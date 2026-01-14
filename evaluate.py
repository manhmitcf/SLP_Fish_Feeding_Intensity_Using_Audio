import torch
import os
from preprocessing.audio_preprocessor import AudioPreprocessor
from preprocessing import config_preprocess as config
from models.swin_transformer import FishSwinTransformer
from evaluation.base_evaluator import BaseEvaluator

def main():
    # --- 1. Configuration ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    # Path to the trained model checkpoint
    CHECKPOINT_PATH = './checkpoints/swin_tiny/best_model.pth'
    
    print(f"Evaluation Configuration:")
    print(f"- Device: {DEVICE}")
    print(f"- Checkpoint: {CHECKPOINT_PATH}")
    
    # --- 2. Data Preparation ---
    print("\n[1/3] Loading Test Data...")
    preprocessor = AudioPreprocessor()
    
    # We only need the test_loader
    _, _, test_loader = preprocessor.get_dataloaders(batch_size=BATCH_SIZE)
    
    if test_loader is None:
        print("Error: Test data not found. Please run preprocessing first.")
        return

    print(f"Test data loaded. Batches: {len(test_loader)}")

    # --- 3. Model Initialization ---
    print("\n[2/3] Initializing Model...")
    # Note: pretrained=False is fine here because we will overwrite weights with the checkpoint
    model = FishSwinTransformer(num_classes=4, pretrained=False) 
    
    # --- 4. Evaluation ---
    print("\n[3/3] Starting Evaluation...")
    evaluator = BaseEvaluator(
        model=model, 
        device=DEVICE, 
        class_mapping=config.CLASS_MAPPING
    )
    
    # Load weights
    if evaluator.load_checkpoint(CHECKPOINT_PATH):
        # Run evaluation
        evaluator.evaluate(test_loader)
    else:
        print("Evaluation aborted due to missing checkpoint.")

if __name__ == "__main__":
    main()
