import torch
import torch.nn as nn
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class BaseTrainer:
    """
    Base class for training loop management.
    Handles the core training logic, validation, and checkpointing.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir):
        """
        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            criterion (Loss): Loss function (e.g., CrossEntropyLoss).
            optimizer (Optimizer): Optimizer (e.g., AdamW).
            device (str): 'cuda' or 'cpu'.
            save_dir (str): Directory to save checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        self.best_acc = 0.0
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train_one_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Move data to device
            inputs = batch['waveform'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                inputs = batch['waveform'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        
        return epoch_loss, epoch_acc

    def fit(self, num_epochs):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save Best Model
            if val_acc > self.best_acc:
                print(f"Validation Accuracy improved ({self.best_acc:.4f} -> {val_acc:.4f}). Saving model...")
                self.best_acc = val_acc
                save_path = os.path.join(self.save_dir, "best_model.pth")
                self.model.save_checkpoint(save_path, self.optimizer, epoch, self.best_acc)
            
            # Optional: Scheduler step could go here (if implemented in subclass)
            self.on_epoch_end(epoch, val_loss)

        print(f"Training complete. Best Validation Accuracy: {self.best_acc:.4f}")

    def on_epoch_end(self, epoch, val_loss):
        """Hook for subclasses to implement scheduler logic."""
        pass
