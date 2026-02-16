import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from physicsnemo.models.fno import FNO
from sonos_dataset import SonosDataset

class SobolevLoss(nn.Module):
    """
    Calculates the H1 Sobolev loss, which includes the MSE of the function values
    and the MSE of the spatial gradients (derivative matching).
    
    L = ||u - u_hat||^2 + alpha * ||grad(u) - grad(u_hat)||^2
    """
    def __init__(self, weight_grad=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.weight_grad = weight_grad

    def forward(self, pred, target):
        # 1. Standard L2 (MSE) Loss on function values
        loss_val = self.mse(pred, target)

        # 2. Compute spatial gradients using central finite differences
        pred_dy, pred_dx = torch.gradient(pred, dim=(-2, -1))
        target_dy, target_dx = torch.gradient(target, dim=(-2, -1))

        loss_dx = self.mse(pred_dx, target_dx)
        loss_dy = self.mse(pred_dy, target_dy)

        # Combine losses
        total_loss = loss_val + self.weight_grad * (loss_dx + loss_dy)
        return total_loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--sobolev_weight", type=float, default=1.0, 
                        help="Weight for the derivative matching loss (PINO/Sobolev regularization).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on device: {device}")
    print(f"Sobolev Loss Weight: {args.sobolev_weight}")

    # 1. Dataset
    # Resolution (64, 128) -> (R, X)
    full_ds = SonosDataset(data_dir=args.data_dir, resolution=(64, 128))
    if len(full_ds) == 0: 
        print("No data found. Exiting.")
        return

    # Save stats
    torch.save(full_ds.get_stats(), os.path.join(args.out_dir, "stats.pt"))

    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model
    # in_channels=6 (Spacing, TL, PGM, Time, X_grid, R_grid)
    model = FNO(
        in_channels=6, 
        out_channels=1, 
        decoder_layers=1, 
        decoder_layer_size=32, 
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=12,
        padding=8
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    
    criterion = SobolevLoss(weight_grad=args.sobolev_weight)

    print("Starting training...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            
            # Calculate Sobolev Loss (Value + Derivatives)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1} | Train (Sobolev): {train_loss:.5f} | Val (Sobolev): {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

    print("Training Complete.")

if __name__ == "__main__":
    train()