import torch
import torch.nn as nn
import torch.optim as optim
from .base_trainer import BaseTrainer

class ResNetTrainer(BaseTrainer):
    """
    Trainer specifically configured for ResNet.
    """
    def __init__(self, model, train_loader, val_loader, device, save_dir, 
                 learning_rate=1e-4, weight_decay=1e-4):
        
        criterion = nn.CrossEntropyLoss()
        
        # ResNet often works well with SGD, but AdamW is also fine and converges faster.
        # Let's stick to AdamW for consistency, or you can switch to SGD:
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device, save_dir)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    def on_epoch_end(self, epoch, val_loss):
        self.scheduler.step(val_loss)
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")