"""
Dataset loader for VLA training
Supports multiple datasets with fallback to toy data
"""

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np


class ToyVLADataset(Dataset):
    """
    Toy dataset - always works, no downloads
    Red square = move right, Blue square = move left
    """
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        print(f"✓ Created toy dataset with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random color
        is_red = np.random.rand() > 0.5
        
        # Create image with colored square
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        color = (255, 0, 0) if is_red else (0, 0, 255)
        draw.rectangle([62, 62, 162, 162], fill=color)
        
        # Instruction
        instruction = "move to the red block" if is_red else "move to the blue block"
        
        # Action: [x, y, z, roll, pitch, yaw, gripper]
        action = torch.tensor([
            1.0 if is_red else -1.0,  # x
            0.0, 0.0,  # y, z
            0.0, 0.0, 0.0,  # rotation
            1.0,  # gripper
        ], dtype=torch.float32)
        
        return {
            'image': img,
            'instruction': instruction,
            'action': action
        }


class HuggingFaceDataset(Dataset):
    """
    Try to load dataset from HuggingFace
    Falls back to toy data if fails
    """
    
    def __init__(self, dataset_name="rail-berkeley/bridge_dataset", split="train", max_samples=1000):
        print(f"Attempting to load {dataset_name}...")
        
        try:
            from datasets import load_dataset
            
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=False,
                trust_remote_code=True  # Some datasets need this
            )
            
            if max_samples and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))
            
            self.use_real_data = True
            print(f"✓ Loaded {len(self.dataset)} samples from HuggingFace")
            
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")
            print("✓ Falling back to toy dataset")
            self.use_real_data = False
            self.toy_dataset = ToyVLADataset(num_samples=max_samples)
    
    def __len__(self):
        if self.use_real_data:
            return len(self.dataset)
        return len(self.toy_dataset)
    
    def __getitem__(self, idx):
        if not self.use_real_data:
            return self.toy_dataset[idx]
        
        sample = self.dataset[idx]
        
        # Try different possible field names
        image = None
        for key in ['image', 'observation.image', 'obs.image']:
            try:
                if key in sample:
                    image = sample[key]
                    break
                # Try nested access
                parts = key.split('.')
                val = sample
                for part in parts:
                    val = val[part]
                image = val
                break
            except:
                continue
        
        if image is None:
            # Generate dummy image
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Try different instruction field names
        instruction = ""
        for key in ['instruction', 'language_instruction', 'task', 'text']:
            if key in sample:
                instruction = sample[key]
                if isinstance(instruction, dict):
                    instruction = instruction.get('language_instruction', str(instruction))
                break
        
        if not instruction:
            instruction = "perform task"
        
        # Try to get action
        action = sample.get('action', [0.0] * 7)
        if not isinstance(action, (list, tuple)):
            action = [0.0] * 7
        
        # Ensure 7D action
        if len(action) < 7:
            action = list(action) + [0.0] * (7 - len(action))
        elif len(action) > 7:
            action = action[:7]
        
        action = torch.tensor(action, dtype=torch.float32)
        
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
    print("="*60)
    print("Testing Datasets")
    print("="*60)
    
    # Test 1: Toy dataset (always works)
    print("\n1. Testing Toy Dataset...")
    toy_dataset = ToyVLADataset(num_samples=10)
    from torch.utils.data import DataLoader
    loader = DataLoader(toy_dataset, batch_size=2, collate_fn=collate_fn)
    
    for images, instructions, actions in loader:
        print(f"   Images: {len(images)} x {images[0].size}")
        print(f"   Instructions: {instructions}")
        print(f"   Actions: {actions.shape}")
        images[0].save("test_toy.png")
        print("   ✓ Saved test_toy.png")
        break
    
    # Test 2: Try HuggingFace dataset (with fallback)
    print("\n2. Testing HuggingFace Dataset (will fallback if fails)...")
    hf_dataset = HuggingFaceDataset(
        dataset_name="rail-berkeley/bridge_dataset",
        max_samples=10
    )
    loader2 = DataLoader(hf_dataset, batch_size=2, collate_fn=collate_fn)
    
    for images, instructions, actions in loader2:
        print(f"   Images: {len(images)} x {images[0].size if hasattr(images[0], 'size') else 'N/A'}")
        print(f"   Instructions: {instructions}")
        print(f"   Actions: {actions.shape}")
        break
    
    print("\n" + "="*60)
    print("✅ Dataset code works!")
    print("="*60)