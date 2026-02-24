"""
VLA-0 Training Script
Supports ToyVLADataset and OpenXDataset
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model import VLA0Model
from dataset import  OpenXDataset, collate_fn


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
    dataset_type="toy",  # "toy" or "openx"
):
    """Train VLA-0"""
    
    print("="*60)
    print("VLA-0 Training (2B Model)")
    print("="*60)
    
    # Create dataset based on type
    if dataset_type == "openx":
        print("\nUsing OpenX-1K dataset...")
        dataset = OpenXDataset(split="train", max_samples=num_samples)
   
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Single thread for simplicity
    )
    
    # Compute action statistics
    action_mean, action_std = compute_dataset_stats(dataloader)
    
    # Create model
    print("\nInitializing model...")
    model = VLA0Model(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        action_dim=7,
        num_bins=256,
        horizon=1,
    )
    
    # Set action statistics
    model.action_mean = action_mean.to(model.model.device)
    model.action_std = action_std.to(model.model.device)
    
    # Optimizer (only train VLM parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )
    
    # Print training info
    print("\n" + "="*60)
    print("Training Configuration:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Steps per epoch: {len(dataloader)}")
    print(f"  Total steps: {len(dataloader) * epochs}")
    print("="*60 + "\n")
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, instructions, actions in pbar:
            optimizer.zero_grad()
            
            # Move actions to device
            actions = actions.to(model.model.device)
            
            # Forward pass
            loss = model(images, instructions, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/num_batches:.4f}'
            })
        
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}")
    
    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'action_mean': action_mean,
        'action_std': action_std,
        'config': {
            'model_name': "Qwen/Qwen2-VL-2B-Instruct",
            'action_dim': 7,
            'num_bins': 256,
            'horizon': 1,
        }
    }
    torch.save(save_dict, "vla0_model.pt")
    print("✅ Model saved to vla0_model.pt")
    print("="*60)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLA-0 model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "openx"], 
                        help="Dataset to use: 'toy' (synthetic) or 'openx' (real robot data)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VLA-0 Training Script")
    print("="*60)
    print(f"Arguments:")
    print(f"  --epochs {args.epochs}")
    print(f"  --batch_size {args.batch_size}")
    print(f"  --lr {args.lr}")
    print(f"  --num_samples {args.num_samples}")
    print(f"  --dataset {args.dataset}")
    print("="*60 + "\n")
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        dataset_type=args.dataset,
    )