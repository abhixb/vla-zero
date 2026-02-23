"""
Simple dataset using Bridge (real robot data, easy to load)
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset


class BridgeVLADataset(Dataset):
    """Use Bridge dataset from HuggingFace"""
    
    def __init__(self, split="train", max_samples=1000):
        """
        Load Bridge dataset - real robot manipulation data
        Super easy, just downloads from HuggingFace
        """
        print(f"Loading Bridge dataset ({split})...")
        
        # Load from HuggingFace (this is the whole dataset code!)
        self.dataset = load_dataset(
            "rail-berkeley/bridge_dataset",
            split=split,
            streaming=False  # Download everything
        )
        
        # Limit samples for faster iteration
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"✓ Loaded {len(self.dataset)} samples")
        
        # Bridge actions are 7D: [x, y, z, roll, pitch, yaw, gripper]
        self.action_dim = 7
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Extract data (Bridge format)
        image = sample['image']  # PIL Image
        instruction = sample['instruction']  # Text
        action = torch.tensor(sample['action'], dtype=torch.float32)  # 7D action
        
        return {
            'image': image,
            'instruction': instruction,
            'action': action
        }


def collate_fn(batch):
    """Collate function for dataloader"""
    images = [item['image'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    
    return images, instructions, actions


# Test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("Testing Bridge dataset...")
    
    # Load just 10 samples for testing
    dataset = BridgeVLADataset(split="train", max_samples=10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    for images, instructions, actions in dataloader:
        print(f"\nBatch:")
        print(f"  Images: {len(images)} x {images[0].size}")
        print(f"  Instructions: {instructions}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Example action: {actions[0]}")
        break
    
    print("\n✅ Bridge dataset works!")