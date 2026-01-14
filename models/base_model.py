import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all models in this project.
    
    Enforces a consistent interface and provides common utility methods
    like saving/loading checkpoints and counting parameters.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Forward pass logic. Must be implemented by subclasses.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output logits.
        """
        pass

    def save_checkpoint(self, path, optimizer=None, epoch=None, best_acc=None):
        """
        Saves the model state dict and optional training info to a file.

        Args:
            path (str): Path to save the checkpoint (.pth file).
            optimizer (Optimizer, optional): Optimizer state to resume training.
            epoch (int, optional): Current epoch number.
            best_acc (float, optional): Best validation accuracy so far.
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        checkpoint = {
            'model_state_dict': self.state_dict(),
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if best_acc is not None:
            checkpoint['best_acc'] = best_acc
            
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """
        Loads model weights from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
            optimizer (Optimizer, optional): Optimizer to load state into.

        Returns:
            dict: The full checkpoint dictionary (useful for retrieving epoch/best_acc).
            None: If file not found.
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}")
            return None
            
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {path}")
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
            
        return checkpoint

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """
        Optional: Freeze all layers except the head.
        Useful for transfer learning.
        """
        for param in self.parameters():
            param.requires_grad = False
        print("All layers frozen.")

    def unfreeze_all(self):
        """
        Unfreeze all layers for fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen.")
