"""
Dataset loader for VLA training
Uses OpenX-Embodiment 1K subset (small, ~1-2GB)
"""

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np


class OpenXDataset(Dataset):
    """
    OpenX-Embodiment 1K Dataset
    Small subset perfect for training (~1-2GB)
    """
    
    def __init__(self, split="train", max_samples=1000):
        print(f"Loading OpenX-1K dataset ({split})...")
        
        try:
            from datasets import load_dataset
            
            # Try multiple possible dataset names/paths
            dataset_names = [
                "jxu124/OpenX-Embodiment",  # Main OpenX repo
                "openx/openx_embodiment",   # Alternative
                  
            ]
            
            self.dataset = None
            for name in dataset_names:
                try:
                    print(f"  Trying {name}...")
                    self.dataset = load_dataset(
                        name,
                        split=split,
                        streaming=False,
                        trust_remote_code=True
                    )
                    print(f"  ✓ Success!")
                    break
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)[:50]}...")
                    continue
            
            if self.dataset is None:
                raise Exception("Could not load any dataset")
            
            # Limit samples
            original_len = len(self.dataset)
            if max_samples and original_len > max_samples:
                self.dataset = self.dataset.select(range(max_samples))
            
            self.use_real_data = True
            print(f"✓ Loaded {len(self.dataset)} samples (from {original_len} total)")
            
        except Exception as e:
            print(f"✗ Failed to load OpenX dataset: {e}")
            print("✓ Falling back to toy dataset (you can still train!)")
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
        
        # Extract image (try multiple field names)
        image = None
        image_keys = ['image', 'observation', 'obs', 'rgb']
        for key in image_keys:
            try:
                val = sample.get(key)
                if val is not None:
                    # Handle nested structures
                    if isinstance(val, dict):
                        image = val.get('image') or val.get('rgb')
                    else:
                        image = val
                    if image is not None:
                        break
            except:
                continue
        
        # Convert to PIL if needed
        if image is None:
            image = Image.new('RGB', (224, 224), color='gray')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Resize to consistent size
        image = image.resize((224, 224))
        
        # Extract instruction
        instruction = ""
        instruction_keys = ['instruction', 'language_instruction', 'task', 'language']
        for key in instruction_keys:
            try:
                val = sample.get(key)
                if val:
                    if isinstance(val, dict):
                        instruction = val.get('language_instruction') or val.get('instruction', '')
                    else:
                        instruction = str(val)
                    if instruction:
                        break
            except:
                continue
        
        if not instruction:
            instruction = "perform manipulation task"
        
        # Extract action
        action = sample.get('action', [0.0] * 7)
        
        # Handle different action formats
        if isinstance(action, dict):
            action = action.get('action', [0.0] * 7)
        
        if not isinstance(action, (list, tuple, np.ndarray)):
            action = [0.0] * 7
        
        # Convert to list if numpy
        if isinstance(action, np.ndarray):
            action = action.flatten().tolist()
        
        # Ensure 7D
        action = list(action)
        if len(action) < 7:
            action = action + [0.0] * (7 - len(action))
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
    from torch.utils.data import DataLoader
    
    print("="*60)
    print("Testing OpenX-1K Dataset")
    print("="*60)
    
    # Try to load OpenX (will fallback to toy if fails)
    dataset = OpenXDataset(split="train", max_samples=10)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    print("\nTesting dataloader...")
    for images, instructions, actions in loader:
        print(f"\nBatch loaded successfully:")
        print(f"  Images: {len(images)} x {images[0].size}")
        print(f"  Instructions: {instructions}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Sample action: {actions[0]}")
        
        # Save sample
        images[0].save("test_openx_sample.png")
        print(f"  ✓ Saved test_openx_sample.png")
        break
    
    print("\n" + "="*60)
    print("✅ Dataset ready!")
    print("="*60)
    
    if dataset.use_real_data:
        print("\n🎉 Using real OpenX data!")
    else:
        print("\n⚠️  Using toy data (OpenX download failed)")
        print("   You can still train! Toy data works fine for testing.")