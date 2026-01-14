import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import sys
import os

class BaseEvaluator:
    """
    Base class for evaluating models.
    Provides methods to load checkpoints, run inference, and calculate metrics.
    """
    def __init__(self, model, device, class_mapping=None):
        """
        Args:
            model (nn.Module): The model architecture.
            device (str): 'cuda' or 'cpu'.
            class_mapping (dict, optional): Mapping from class names to integers.
                                            Used for pretty printing reports.
        """
        self.model = model
        self.device = device
        self.class_mapping = class_mapping
        
        # Reverse mapping for display (0 -> 'none')
        if self.class_mapping:
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
            self.target_names = [self.idx_to_class[i] for i in range(len(self.class_mapping))]
        else:
            self.target_names = None

    def load_checkpoint(self, checkpoint_path):
        """
        Loads model weights from a checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return False
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle case where checkpoint saves 'model_state_dict' or just state_dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"Successfully loaded weights from {checkpoint_path}")
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def evaluate(self, data_loader):
        """
        Runs evaluation on the provided data loader.
        
        Args:
            data_loader (DataLoader): The test data loader.
            
        Returns:
            dict: A dictionary containing accuracy, report, and confusion matrix.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        print("Running Inference...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                inputs = batch['waveform'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate Metrics
        acc = accuracy_score(all_targets, all_preds)
        
        # Detailed Report
        report = classification_report(
            all_targets, 
            all_preds, 
            target_names=self.target_names,
            digits=4
        )
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        results = {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        self._print_results(results)
        
        return results

    def _print_results(self, results):
        """Helper to print evaluation results nicely."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print("-" * 50)
        print("Classification Report:")
        print(results['classification_report'])
        print("-" * 50)
        print("Confusion Matrix:")
        print(results['confusion_matrix'])
        print("="*50)
