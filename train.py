import torch
import numpy as np
import random
import os
from preprocessing.audio_preprocessor import AudioPreprocessor
from models.swin_transformer import FishSwinTransformer
from trainers.swin_trainer import SwinTrainer

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # --- 1. Configuration ---
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = './checkpoints/swin_tiny'
    
    print(f"Training Configuration:")
    print(f"- Device: {DEVICE}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Epochs: {NUM_EPOCHS}")
    print(f"- Learning Rate: {LEARNING_RATE}")
    print(f"- Save Directory: {SAVE_DIR}")
    
    set_seed(SEED)

    # --- 2. Data Preparation ---
    print("\n[1/3] Preparing Data...")
    preprocessor = AudioPreprocessor()
    
    # Try to get dataloaders
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=BATCH_SIZE)
    
    # If data is missing (loaders are None), run preprocessing automatically
    if train_loader is None:
        print("Processed data not found. Running preprocessing pipeline (split_and_save)...")
        preprocessor.split_and_save()
        # Reload dataloaders after processing
        train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=BATCH_SIZE)
        
        if train_loader is None:
             print("Error: Failed to create DataLoaders even after preprocessing. Check RAW_DATA_PATH.")
             return

    print(f"Data loaded successfully.")
    print(f"- Train batches: {len(train_loader)}")
    print(f"- Val batches: {len(val_loader)}")

    # --- 3. Model Initialization ---
    print("\n[2/3] Initializing Model...")
    model = FishSwinTransformer(num_classes=4, pretrained=True)
    model.to(DEVICE)
    
    # Optional: Print model summary
    print(f"Model created: Swin Transformer Tiny")
    print(f"Trainable Parameters: {model.count_parameters():,}")

    # --- 4. Training ---
    print("\n[3/3] Starting Training...")
    trainer = SwinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        save_dir=SAVE_DIR,
        learning_rate=LEARNING_RATE
    )
    
    trainer.fit(num_epochs=NUM_EPOCHS)
    
    print("\nTraining finished successfully!")
    print(f"Best model saved to: {os.path.join(SAVE_DIR, 'best_model.pth')}")
    print("You can now run 'python evaluate.py' to test the model.")

if __name__ == "__main__":
    main()
