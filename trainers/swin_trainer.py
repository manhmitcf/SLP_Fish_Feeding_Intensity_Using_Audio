import torch
import torch.nn as nn
import torch.optim as optim
from .base_trainer import BaseTrainer

class SwinTrainer(BaseTrainer):
    """
    Trainer specifically configured for Swin Transformer.
    Uses AdamW optimizer and CosineAnnealingLR scheduler.
    """
    def __init__(self, model, train_loader, val_loader, device, save_dir, 
                 learning_rate=1e-4, weight_decay=1e-4):
        
        # 1. Define Loss Function
        criterion = nn.CrossEntropyLoss()
        
        # 2. Define Optimizer (AdamW is best for Transformers)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Initialize BaseTrainer
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device, save_dir)
        
        # 3. Define Scheduler (Cosine Annealing)
        # T_max is usually set to the number of epochs
        # We'll set it dynamically in fit() or assume a default here.
        # For now, let's initialize it but we might need to step it carefully.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    def on_epoch_end(self, epoch, val_loss):
        """
        Called at the end of each epoch.
        Updates the learning rate scheduler based on validation loss.
        """
        # Step the scheduler
        self.scheduler.step(val_loss)
        
        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
