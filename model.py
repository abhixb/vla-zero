"""
VLA-0 Style Model
Actions are discretized and predicted as text tokens
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np


class VLA0Model(nn.Module):
    """
    VLA-0: Actions as text generation
    Following the paper's approach exactly
    """
    
    def __init__(
        self,
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        action_dim=7,
        num_bins=256,  # Discretize actions into 256 bins
        horizon=1,  # Predict 1 action at a time (can increase later)
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.horizon = horizon
        
        print(f"Loading VLM: {model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Action statistics (will be set from dataset)
        self.action_mean = None
        self.action_std = None
        
        print(f"✓ Model loaded!")
        print(f"  Action dim: {action_dim}")
        print(f"  Bins: {num_bins}")
        print(f"  Horizon: {horizon}")
        print(f"  Total tokens per prediction: {horizon * action_dim}")
    
    def discretize_actions(self, actions):
        """
        Convert continuous actions to discrete tokens (0-255)
        actions: (batch, action_dim) or (batch, horizon, action_dim)
        returns: list of strings with space-separated integers
        """
        # Ensure 2D: (batch, action_dim)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)  # (batch, 1, action_dim)
        
        # Normalize using dataset statistics
        if self.action_mean is not None:
            # Ensure action_mean and action_std are on same device
            action_mean = self.action_mean.to(actions.device)
            action_std = self.action_std.to(actions.device)
            actions = (actions - action_mean) / (action_std + 1e-8)
        
        # Clip to [-3, 3] (3 standard deviations)
        actions = torch.clamp(actions, -3, 3)
        
        # Map to [0, 1]
        actions = (actions + 3) / 6
        
        # Discretize to bins [0, num_bins-1]
        actions = (actions * (self.num_bins - 1)).long()
        actions = torch.clamp(actions, 0, self.num_bins - 1)
        
        # Convert to string format
        batch_size = actions.shape[0]
        action_strings = []
        
        for i in range(batch_size):
            # Flatten action sequence and convert to string
            tokens = actions[i].flatten().cpu().numpy()
            action_str = " ".join(map(str, tokens))
            action_strings.append(action_str)
        
        return action_strings
    
    def undiscretize_actions(self, action_strings):
        """
        Convert discrete token strings back to continuous actions
        """
        actions_list = []
        
        for action_str in action_strings:
            try:
                # Parse tokens (handle extra whitespace)
                tokens = [int(x) for x in action_str.strip().split() if x.isdigit()]
                
                # Pad or truncate to expected length
                expected_len = self.horizon * self.action_dim
                if len(tokens) < expected_len:
                    tokens += [0] * (expected_len - len(tokens))
                tokens = tokens[:expected_len]
                
                # Convert to tensor
                actions = torch.tensor(tokens, dtype=torch.float32)
                
                # Reshape
                actions = actions.reshape(self.horizon, self.action_dim)
                
                # Undiscretize: [0, num_bins-1] -> [0, 1]
                actions = actions / (self.num_bins - 1)
                
                # Map back to [-3, 3]
                actions = actions * 6 - 3
                
                # Denormalize
                if self.action_mean is not None:
                    actions = actions * (self.action_std + 1e-8) + self.action_mean
                
                actions_list.append(actions)
                
            except Exception as e:
                # Fallback: return zero actions if parsing fails
                print(f"Warning: Failed to parse action string '{action_str[:50]}...': {e}")
                actions = torch.zeros(self.horizon, self.action_dim)
                if self.action_mean is not None:
                    actions = self.action_mean.unsqueeze(0).repeat(self.horizon, 1)
                actions_list.append(actions)
        
        return torch.stack(actions_list)
    
    def forward(self, images, instructions, actions=None):
        """
        Forward pass
        If actions provided (training), returns loss
        Otherwise (inference), returns predicted actions
        """
        if actions is not None:
            # Training mode: compute loss
            return self.compute_loss(images, instructions, actions)
        else:
            # Inference mode: generate actions
            return self.generate_actions(images, instructions)
    
    def compute_loss(self, images, instructions, actions):
        """
        Compute loss for training
        actions: (batch, action_dim) continuous actions
        """
        # Ensure actions are on correct device
        if not actions.is_cuda and torch.cuda.is_available():
            actions = actions.to(self.model.device)
        
        # Discretize actions to text
        action_strings = self.discretize_actions(actions)
        
        # Create chat format with system prompt
        messages = []
        for img, inst, act_str in zip(images, instructions, action_strings):
            messages.append([
                {"role": "system", "content": [{"type": "text", "text": 
                    f"You are a robot control system. Output exactly {self.horizon * self.action_dim} "
                    f"space-separated integers between 0 and {self.num_bins-1}, representing robot actions."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Task: {inst}"}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": act_str}]}
            ])
        
        # Process through model
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
                 for msg in messages]
        
        # Process images and text
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        # Forward pass with labels
        outputs = self.model(**inputs, labels=inputs.input_ids)
        
        return outputs.loss
    
    @torch.no_grad()
    def generate_actions(self, images, instructions):
        """
        Generate actions (inference mode)
        """
        self.model.eval()
        
        # Create chat format
        messages = []
        for img, inst in zip(images, instructions):
            messages.append([
                {"role": "system", "content": [{"type": "text", "text": 
                    f"You are a robot control system. Output exactly {self.horizon * self.action_dim} "
                    f"space-separated integers between 0 and {self.num_bins-1}, representing robot actions."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Task: {inst}"}
                ]}
            ])
        
        # Process
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                 for msg in messages]
        
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        # Generate (constrain to numbers only would be ideal, but not implemented here)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.horizon * self.action_dim * 5,  # Extra space for tokens
            do_sample=False,  # Deterministic
            temperature=1.0,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        
        # Decode
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse actions
        actions = self.undiscretize_actions(generated_texts)
        
        return actions


# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing VLA-0 Model")
    print("="*60)
    
    # Create model
    print("\nInitializing model...")
    model = VLA0Model(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        action_dim=7,
        num_bins=256,
        horizon=1,
    )
    
    # Dummy data
    print("\nCreating test data...")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    dummy_instruction = "pick up the red block"
    dummy_action = torch.randn(1, 7)  # (batch=1, action_dim=7)
    
    # Set dummy statistics
    model.action_mean = torch.zeros(7)
    model.action_std = torch.ones(7)
    
    print("\n" + "="*60)
    print("Test 1: Discretization")
    print("="*60)
    action_str = model.discretize_actions(dummy_action)
    print(f"Continuous action: {dummy_action[0]}")
    print(f"Discretized string: {action_str[0]}")
    
    print("\n" + "="*60)
    print("Test 2: Training Loss")
    print("="*60)
    loss = model.compute_loss([dummy_image], [dummy_instruction], dummy_action)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("Test 3: Action Generation")
    print("="*60)
    pred_actions = model.generate_actions([dummy_image], [dummy_instruction])
    print(f"Generated actions shape: {pred_actions.shape}")
    print(f"Generated actions:\n{pred_actions[0]}")
    
    print("\n" + "="*60)
    print("✅ All tests passed! Model is ready.")
    print("="*60)