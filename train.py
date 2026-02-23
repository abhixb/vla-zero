"""
VLA-0 Training Script
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from model import VLA0Model
from dataset import BridgeVLADataset, collate_fn  # or ToyVLADataset


def compute_dataset_stats(dataloader):
    """Compute mean and std of actions in dataset"""
    print("Computing dataset statistics...")
    all_actions = []
    
    for images, instructions, actions in tqdm(dataloader, desc="Stats"):
        all_actions.append(actions)
    
    all_actions = torch.cat(all_actions, dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0)
    
    print(f"Action mean: {mean}")
    print(f"Action std: {std}")
    
    return mean, std


def train(
    epochs=10,
    batch_size=4,
    lr=1e-5,
    num_samples=1000,
    dataset_type="bridge",  # "bridge" or "toy"
):
    """Train VLA-0"""
    
    print("="*60)
    print("VLA-0 Training")
    print("="*60)
    
    # Create dataset
    if dataset_type == "bridge":
        from dataset import BridgeVLADataset
        dataset = BridgeVLADataset(split="train", max_samples=num_samples)
    else:
        from dataset import ToyVLADataset
        dataset = ToyVLADataset(num_samples=num_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Compute action statistics
    action_mean, action_std = compute_dataset_stats(dataloader)
    
    # Create model
    model = VLA0Model(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        action_dim=7,
        num_bins=256,
        horizon=1,
    )
    
    # Set action statistics
    model.action_mean = action_mean.to(model.model.device)
    model.action_std = action_std.to(model.model.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, instructions, actions in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(images, instructions, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    save_dict = {
        'model': model.state_dict(),
        'action_mean': action_mean,
        'action_std': action_std,
    }
    torch.save(save_dict, "vla0_model.pt")
    print("✅ Model saved to vla0_model.pt")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="toy", choices=["bridge", "toy"])
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        dataset_type=args.dataset,
    )