import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class OpenVLADataset(Dataset):
    def __init__(self, split="train", max_samples=1000):
        """
        Load OpenVLA-1K dataset (tttonyalpha/openvla_1k-dataset).
        This dataset is a 1.4k subset ideal for VLA fine-tuning.
        """
        print(f"Loading OpenVLA-1K dataset ({split})...")
        
        self.dataset = load_dataset(
            "tttonyalpha/openvla_1k-dataset",
            split=split,
            streaming=False
        )
        
        if max_samples:
            total_len = len(self.dataset)
            self.dataset = self.dataset.select(range(min(max_samples, total_len)))
            
        print(f"✓ Loaded {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Handle common robot dataset nesting (observation vs flat)
        image = sample.get('image') or sample.get('observation', {}).get('image')
        instruction = sample.get('instruction') or sample.get('task', {}).get('language_instruction', "")
        
        # Action is usually a 7D list: [x, y, z, roll, pitch, yaw, gripper]
        action_data = sample.get('action', [0.0] * 7)
        action = torch.tensor(action_data, dtype=torch.float32)
        
        return {
            'image': image, 
            'instruction': instruction, 
            'action': action
        }

def collate_fn(batch):
    """
    Improved logic: Group by key first, then stack or listify.
    This avoids syntax errors and handles different data types safely.
    """
    # Create a dictionary of lists from the list of dictionaries
    # Result: {'image': [...], 'instruction': [...], 'action': [...]}
    data_dict = {key: [item[key] for item in batch] for key in batch[0].keys()}
    
    # Only stack tensors (like actions); keep images and text as lists
    data_dict['action'] = torch.stack(data_dict['action'])
    
    return data_dict

# Test script
if __name__ == "__main__":
    try:
        # Load a small sample
        dataset = OpenVLADataset(split="train", max_samples=10)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        for batch in dataloader:
            print(f"\nBatch Successfully Loaded:")
            print(f"  Images: {len(batch)} samples")
            print(f"  First Image Size: {batch[0].size}")
            print(f"  Instructions: {batch['instruction']}")
            print(f"  Actions Shape: {batch['action'].shape}")
            break
            
        print("\n Dataset logic is now robust and ready!")
        
    except Exception as e:
        print(f"\n Still hitting an error: {e}")
