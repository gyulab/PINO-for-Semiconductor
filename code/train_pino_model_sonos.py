# SPDX-License-Identifier: Apache-2.0
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict
from scipy.interpolate import griddata

import hydra.utils
import physicsnemo.sym
from physicsnemo.models.layers.spectral_layers import fourier_derivatives
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter
from ops import dx, ddx  # Internal PhysicsNeMo FDM Helpers

# ==========================================
# 1. Physics Residual Node (Poisson)
# ==========================================
class Poisson(torch.nn.Module):
    """Custom Poisson PDE definition for SONOS PINO with Variable Geometries"""

    def __init__(self, gradient_method: str = "exact", epsilon_rel: float = 7.5):
        super().__init__()
        self.gradient_method = str(gradient_method)
        # epsilon_0 = 8.854e-12 F/m
        self.epsilon = epsilon_rel * 8.8541878128e-12 

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get tensors (Shape: Batch, Channels, R_dim, X_dim)
        phi = input_var["phi"]
        rho = input_var["rho"]
        
        # Get dynamic geometry sizes for scaling.
        # R_length/X_length are broadcast grids (B,1,H,W) — all spatial values
        # are identical, so slice the [0,0] point and reshape to (B,1,1,1).
        R_len = input_var["R_length"][:, :, 0, 0].view(-1, 1, 1, 1)
        X_len = input_var["X_length"][:, :, 0, 0].view(-1, 1, 1, 1)

        # Base Normalized Spacings (assuming length 1.0)
        dr_norm = 1.0 / (phi.shape[-2] - 1)
        dx_norm = 1.0 / (phi.shape[-1] - 1)

        # Compute 2nd order spatial gradients
        if self.gradient_method == "exact":
            # Exact gradients are calculated on a [1.0, 1.0] domain
            # FNO PINO uses x/y/z axis names internally (r->x, x->y)
            # We divide by length^2 to map to the true physical domain size
            d2phidr2 = input_var["phi__x__x"] / (R_len ** 2)
            d2phidx2 = input_var["phi__y__y"] / (X_len ** 2)
            
        elif self.gradient_method == "fdm":
            # Compute FDM assuming domain [1.0, 1.0], then scale
            d2phi_norm_r = ddx(phi, dx=dr_norm, channel=0, dim=0, order=1, padding="replication")
            d2phi_norm_x = ddx(phi, dx=dx_norm, channel=0, dim=1, order=1, padding="replication")
            d2phidr2 = d2phi_norm_r / (R_len ** 2)
            d2phidx2 = d2phi_norm_x / (X_len ** 2)
            
        elif self.gradient_method == "fourier":
            dim_r = phi.shape[2]
            dim_x = phi.shape[3]
            phi_padded = F.pad(phi, (0, dim_x - 1, 0, dim_r - 1), mode="reflect")
            
            # Request fourier derivatives over [1.0, 1.0] domain
            _, f_d2phi = fourier_derivatives(phi_padded, [1.0, 1.0])
            d2phi_norm_r = f_d2phi[:, 0:1, :dim_r, :dim_x]
            d2phi_norm_x = f_d2phi[:, 1:2, :dim_r, :dim_x]
            
            d2phidr2 = d2phi_norm_r / (R_len ** 2)
            d2phidx2 = d2phi_norm_x / (X_len ** 2)
        else:
            raise ValueError(f"Derivative method {self.gradient_method} not supported.")

        # POISSON EQUATION: \epsilon * (\nabla^2 \phi) + \rho = 0
        poisson_pde = self.epsilon * (d2phidr2 + d2phidx2) + rho

        # Zero outer boundary to avoid edge artifacts from differentiation
        poisson_pde = F.pad(poisson_pde[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        
        # Scale to O(1) so PyTorch Optimizer handles it safely
        scaling_factor = 1e-15 
        output_var = {"poisson": poisson_pde * scaling_factor} 
        return output_var

# ==========================================
# 2. Data Parsing Code -- build your own data loader here
# ==========================================

# ==========================================
# 3. Main PINO Execution
# ==========================================
@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Set FNO input keys - FNO now natively reads spatial mesh grids 
    # to understand variations in dataset geometry boundaries
    input_keys = [Key("rho"), Key("r_coord"), Key("x_coord")]
    output_keys = [Key("phi")]

    print("Preparing Variable Geometries...")

    original_cwd = hydra.utils.get_original_cwd()
    data_dir = os.path.join(original_cwd, "data")
    invar, outvar = prepare_sonos_data(data_dir=data_dir, grid_res=(64, 128))
                                       
    # Split into Train and Validation sets (80 / 20 split)
    total_snaps = len(invar["rho"])
    split_idx   = int(total_snaps * 0.8)
    train_idx   = list(range(0, split_idx))
    test_idx    = list(range(split_idx, total_snaps))

    invar_train  = {k: v[train_idx] for k, v in invar.items()}
    outvar_train = {k: v[train_idx] for k, v in outvar.items()}
    outvar_train["poisson"] = np.zeros_like(outvar_train["phi"])  # PDE residual target = 0

    invar_test  = {k: v[test_idx] for k, v in invar.items()}
    outvar_test = {k: v[test_idx] for k, v in outvar.items()}

    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset  = DictGridDataset(invar_test,  outvar_test)

    # Build FNO Model
    decoder_net = instantiate_arch(cfg=cfg.arch.decoder, output_keys=output_keys)
    fno = instantiate_arch(cfg=cfg.arch.fno, input_keys=input_keys, decoder_net=decoder_net)

    if cfg.custom.gradient_method == "exact":
        # Force analytical derivative extraction on a normalized 1.0×1.0 square domain.
        # FNO PINO only accepts x/y/z axis names (hardcoded in fno.py line 534).
        # r-axis maps to 'x', x-axis maps to 'y' in FNO's internal convention.
        derivatives = [
            Key("phi", derivatives=[Key("x"), Key("x")]),  # d2phi/dr2 (r -> FNO 'x')
            Key("phi", derivatives=[Key("y"), Key("y")]),  # d2phi/dx2 (x -> FNO 'y')
        ]
        fno.add_pino_gradients(derivatives=derivatives, domain_length=[1.0, 1.0])

    # Connect Nodes
    inputs = ["phi", "rho", "R_length", "X_length"]
    if cfg.custom.gradient_method == "exact":
        inputs += ["phi__x__x", "phi__y__y"]  # FNO axis names: r->x, x->y
        
    poisson_node = Node(
        inputs=inputs,
        outputs=["poisson"],
        evaluate=Poisson(gradient_method=cfg.custom.gradient_method, epsilon_rel=7.5),
        name="Poisson Node",
    )
    
    nodes = [fno.make_node("fno"), poisson_node]

    # Constrain the Domain
    domain = Domain()
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")

    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=2),
        requires_grad=True, 
    )
    domain.add_validator(val, "test")

    # Start training
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
