"""
inference_pino_time.py
----------------------
Time-sweep inference for the PINO SONOS model trained with pino_sonos.py.

Loads the discrete DCD snapshots for a given geometry, then generates
smooth intermediate frames by linearly interpolating the rho input grids
between consecutive snapshots. The PINO model is queried at every
interpolated frame — demonstrating generalisation beyond training data.

Usage
-----
    python inference_pino_time.py [--data_dir ./data]
                                  [--checkpoint ./outputs/pino_sonos/fno.0.pth]
                                  [--spacing 25] [--tl 30] [--pgm 15]
                                  [--n_frames 60] [--output sonos_pino_time.gif]
"""

import os
import argparse
import shutil
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.interpolate import griddata

from pino_sonos import get_stem


def parse_ginestra_csv_with_times(filepath):
    """Like parse_ginestra_csv but also returns the list of timestamps (seconds)."""
    import numpy as np
    with open(filepath, 'r') as f:
        lines = f.readlines()

    snapshots, timestamps = [], []
    current_data = []
    current_time = None
    in_data = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#HEADER'):
            in_data = False
            current_time = None
            continue
        if line.startswith('Time (s),'):
            current_time = float(line.split(',')[1])
            continue
        if line == '#DATA':
            in_data = True
            current_data = []
            continue
        if line.startswith('#'):
            if in_data and current_data:
                snapshots.append(np.array(current_data))
                timestamps.append(current_time if current_time is not None else float('nan'))
                in_data = False
            current_time = None
            continue
        if in_data:
            if 'Ryz' in line:
                continue
            current_data.append([float(x) for x in line.split(',')])

    if in_data and current_data:
        snapshots.append(np.array(current_data))
        timestamps.append(current_time if current_time is not None else float('nan'))

    return snapshots, timestamps


def _fmt_time(t_sec: float) -> str:
    """Return a human-readable time string."""
    if t_sec >= 1000:
        return f"{t_sec:.0f} s"
    elif t_sec >= 10:
        return f"{t_sec:.1f} s"
    elif t_sec >= 0.1:
        return f"{t_sec:.2f} s"
    else:
        return f"{t_sec:.3f} s"


# ── Model loading ────────────────────────────────────────────────────────────

def load_fno_model(checkpoint_path: str, device: torch.device):
    from physicsnemo.sym.models.fno import FNOArch
    from physicsnemo.sym.models.fully_connected import ConvFullyConnectedArch
    from physicsnemo.sym.key import Key

    latent_size = 32
    decoder = ConvFullyConnectedArch(
        input_keys=[Key("z", size=latent_size)],
        output_keys=[Key("phi")],
        layer_size=256,
        nr_layers=4,
    )
    fno = FNOArch(
        input_keys=[Key("rho"), Key("r_coord"), Key("x_coord")],
        dimension=2,
        decoder_net=decoder,
        nr_fno_layers=4,
        fno_modes=16,
        padding=8,
        padding_type="constant",
        coord_features=True,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fno.load_state_dict(state)
    fno.eval()
    print(f"[Model] Loaded: {checkpoint_path}")
    return fno


# ── Helpers ──────────────────────────────────────────────────────────────────

def interpolate_snap(snap: np.ndarray,
                     R_min, R_max, X_min, X_max,
                     H=64, W=128) -> np.ndarray:
    """Scatter-interpolate one Ginestra snapshot → uniform (H, W) rho grid."""
    pts  = snap[:, 0:2]
    vals = snap[:, 2] * 1e6   # C/cm³ → C/m³
    R_uni = np.linspace(R_min, R_max, H)
    X_uni = np.linspace(X_min, X_max, W)
    X_mesh, R_mesh = np.meshgrid(X_uni, R_uni)
    return griddata(pts, vals, (R_mesh, X_mesh), method="cubic", fill_value=0.0)


def infer(model, rho_grid, R_mesh, X_mesh, device):
    """Run one model forward pass; return phi (H,W) numpy array."""
    with torch.no_grad():
        invar = {
            "rho":     torch.tensor(rho_grid[np.newaxis, np.newaxis], dtype=torch.float32).to(device),
            "r_coord": torch.tensor(R_mesh[np.newaxis, np.newaxis],   dtype=torch.float32).to(device),
            "x_coord": torch.tensor(X_mesh[np.newaxis, np.newaxis],   dtype=torch.float32).to(device),
        }
        return model(invar)["phi"].cpu().numpy().squeeze()


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = load_fno_model(args.checkpoint, device)
    H, W = 64, 128

    # ── Locate DCD file
    stem    = f"spacing{int(args.spacing)}nm-tl{int(args.tl)}am-pgm{int(args.pgm)}v"
    pattern = os.path.join(args.data_dir, "defect-charge-density", f"{stem}-dcd.csv")
    matches = glob.glob(pattern)
    if not matches:
        avail = [get_stem(f)
                 for f in glob.glob(os.path.join(args.data_dir, "defect-charge-density", "*.csv"))]
        raise FileNotFoundError(
            f"No DCD file for: {stem}\nAvailable stems:\n" +
            "\n".join(f"  {s}" for s in avail)
        )
    snapshots, snap_times = parse_ginestra_csv_with_times(matches[0])
    N = len(snapshots)
    print(f"[Data] {N} training snapshots  "
          f"(t = {_fmt_time(snap_times[0])} … {_fmt_time(snap_times[-1])})  "
          f"→  will interpolate to {args.n_frames} frames")

    if N < 2:
        raise RuntimeError("Need at least 2 snapshots to interpolate.")

    # ── Fixed physical domain (use first snapshot bounds)
    pts0    = snapshots[0][:, 0:2]
    R_min   = pts0[:, 0].min();  R_max = pts0[:, 0].max()
    X_min   = pts0[:, 1].min();  X_max = pts0[:, 1].max()
    R_uni   = np.linspace(R_min, R_max, H)
    X_uni   = np.linspace(X_min, X_max, W)
    X_mesh, R_mesh = np.meshgrid(X_uni, R_uni)
    print(f"[Domain] R: [{R_min*1e9:.2f}, {R_max*1e9:.2f}] nm  "
          f"X: [{X_min*1e9:.2f}, {X_max*1e9:.2f}] nm")

    # ── Pre-interpolate all training snapshots onto the uniform grid
    print("Interpolating training snapshots to grid...")
    rho_grids = [interpolate_snap(s, R_min, R_max, X_min, X_max, H, W)
                 for s in snapshots]   # list of (H, W) arrays

    # ── Generate n_frames uniformly-spaced query points in [0, N-1]
    #    Each query point t_q ∈ [0, N-1] linearly interpolates between
    #    adjacent training snapshots  rho_i  and  rho_{i+1}.
    t_queries = np.linspace(0, N - 1, args.n_frames)

    print(f"Running PINO inference on {args.n_frames} interpolated frames...")
    phi_frames = []
    rho_frames = []

    for q_idx, tq in enumerate(t_queries):
        i   = int(tq)
        alpha = tq - i          # fractional distance to next snapshot
        if i >= N - 1:
            i, alpha = N - 2, 1.0

        # Linearly interpolate rho input between snapshots i and i+1
        rho_interp = (1.0 - alpha) * rho_grids[i] + alpha * rho_grids[i + 1]

        phi = infer(model, rho_interp, R_mesh, X_mesh, device)
        phi_frames.append(phi)
        rho_frames.append(rho_interp)

        if q_idx % 10 == 0 or q_idx == args.n_frames - 1:
            print(f"  Frame {q_idx+1}/{args.n_frames}  "
                  f"(snap {i:.1f}→{i+1:.1f}, α={alpha:.2f})")

    phi_arr = np.array(phi_frames)
    rho_arr = np.array(rho_frames)

    # ── Locked colour scales
    phi_lim = np.max(np.abs(phi_arr)) * 1.05
    rho_lim = np.percentile(np.abs(rho_arr), 99) * 1.05
    phi_lvl = np.linspace(-phi_lim, phi_lim, 100)
    rho_lvl = np.linspace(-rho_lim, rho_lim, 100)
    print(f"[Scale] φ ±{phi_lim:.3e} V   ρ ±{rho_lim:.3e} C/m³")

    # ── Render
    temp_dir = "pino_time_frames"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    Xnm = X_mesh * 1e9
    Rnm = R_mesh * 1e9

    for fi in range(args.n_frames):
        tq    = t_queries[fi]
        i     = int(tq)
        alpha = tq - i
        if i >= N - 1:
            i, alpha = N - 2, 1.0
        # Interpolated physical time
        t_interp = (1.0 - alpha) * snap_times[i] + alpha * snap_times[i + 1]
        time_str  = _fmt_time(t_interp)
        label_str = (f"Interp: snap {i+1}→{i+2}  α={alpha:.2f}  t = {time_str}"
                     if abs(alpha) > 1e-6
                     else f"Snapshot {i+1}  t = {time_str}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cf0 = axes[0].contourf(Xnm, Rnm, phi_arr[fi], levels=phi_lvl,
                               cmap="RdBu_r", vmin=-phi_lim, vmax=phi_lim)
        fig.colorbar(cf0, ax=axes[0], label="φ (V)")
        axes[0].set_title(f"Predicted Potential φ  |  {label_str}")
        axes[0].set_xlabel("X (nm)"); axes[0].set_ylabel("R (nm)")

        cf1 = axes[1].contourf(Xnm, Rnm, rho_arr[fi], levels=rho_lvl,
                               cmap="seismic", vmin=-rho_lim, vmax=rho_lim)
        fig.colorbar(cf1, ax=axes[1], label="ρ (C/m³)")
        axes[1].set_title(f"Interpolated Input ρ  |  "
                          f"S={args.spacing}nm, TL={args.tl}Å, PGM={args.pgm}V")
        axes[1].set_xlabel("X (nm)"); axes[1].set_ylabel("R (nm)")

        plt.suptitle(f"PINO SONOS — Time Sweep  t = {time_str}", fontsize=13)
        plt.tight_layout()
        fname = os.path.join(temp_dir, f"frame_{fi:04d}.png")
        plt.savefig(fname, dpi=100)
        plt.close(fig)

    # ── Compile GIF
    out_path = args.output
    print(f"Compiling GIF → {out_path} ...")
    with imageio.get_writer(out_path, mode="I", duration=0.12) as writer:
        for f in sorted(glob.glob(os.path.join(temp_dir, "*.png"))):
            writer.append_data(imageio.imread(f))

    shutil.rmtree(temp_dir)
    print(f"Done!  {args.n_frames} frames saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Time-sweep PINO inference with interpolation between snapshots"
    )
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--checkpoint", type=str,   default="./outputs/pino_sonos/fno.0.pth")
    parser.add_argument("--spacing",    type=float, default=25.0,  help="Spacing (nm)")
    parser.add_argument("--tl",         type=float, default=30.0,  help="Tunnel layer (Å)")
    parser.add_argument("--pgm",        type=float, default=15.0,  help="Program voltage (V)")
    parser.add_argument("--n_frames",   type=int,   default=60,    help="Total frames in GIF")
    parser.add_argument("--output",     type=str,   default="sonos_pino_time.gif")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
