"""
Updated dataset using OpenVLA-1K (tttonyalpha/openvla_1k-dataset)
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset


class OpenVLADataset(Dataset):
    """Use OpenVLA-1K dataset from HuggingFace"""
    
    def __init__(self, split="train", max_samples=1000):
        """
        Load OpenVLA-1K dataset - curated for VLA fine-tuning
        Dataset ID: tttonyalpha/openvla_1k-dataset
        """
        print(f"Loading OpenVLA-1K dataset ({split})...")
        
        # Load the curated 1.4k episode dataset
        # This dataset is designed to help VLA models align with new instructions
        self.dataset = load_dataset(
            "tttonyalpha/openvla_1k-dataset",
            split=split,
            streaming=False
        )
        
        # Limit samples for faster iteration/testing
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"✓ Loaded {len(self.dataset)} samples")
        
        # OpenVLA standard actions are typically 7-DoF: [x, y, z, roll, pitch, yaw, gripper]
        self.action_dim = 7
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Extract features (adjusting for common HuggingFace robot dataset schemas)
        # Often these are stored as 'image', 'instruction', and 'action'
        image = sample if 'image' in sample else sample['observation']
        instruction = sample['instruction'] if 'instruction' in sample else sample['task']['language_instruction']
        
        # Convert action to 7D float tensor
        action_data = sample['action']
        action = torch.tensor(action_data, dtype=torch.float32)
        
        return {
            'image': image,
            'instruction': instruction,
            'action': action
        }


def collate_fn(batch):
    """Collate function for dataloader to handle PIL images and variable text"""
    images = for item in batch]
    instructions = [item['instruction'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    
    return images, instructions, actions


# Test script
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("Testing OpenVLA-1K dataset...")
    
    # Test with a small batch
    try:
        dataset = OpenVLADataset(split="train", max_samples=10)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        for images, instructions, actions in dataloader:
            print(f"\nBatch successfully loaded:")
            print(f"  Images: {len(images)} samples, first size: {images[0].size}")
            print(f"  Instructions: {instructions}")
            print(f"  Actions shape: {actions.shape}")
            print(f"  Example action (7D): {actions[0]}")
            break
            
        print("\n OpenVLA-1K dataset is ready for training!")
        
    except Exception as e:
        print(f"\n Error loading dataset: {e}")
        print("Note: Ensure you have 'datasets' and 'torch' installed.")
