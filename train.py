import os
import sys
import argparse

# --- KAGGLE SECRETS SETUP (MUST BE BEFORE IMPORTS) ---
# Try to load secrets from Kaggle environment and set them as Environment Variables
try:
    from kaggle_secrets import UserSecretsClient
    print("Detected Kaggle environment. Loading secrets...")
    user_secrets = UserSecretsClient()
    
    # Load RAW_DATA_PATH
    try:
        raw_path = user_secrets.get_secret("RAW_DATA_PATH")
        if raw_path:
            os.environ["RAW_DATA_PATH"] = raw_path
            print(f"Set RAW_DATA_PATH from Secrets: {raw_path}")
    except Exception:
        print("Secret 'RAW_DATA_PATH' not found.")

    # Load CUDA_VISIBLE_DEVICES (Optional)
    try:
        cuda_dev = user_secrets.get_secret("CUDA_VISIBLE_DEVICES")
        if cuda_dev:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_dev
            print(f"Set CUDA_VISIBLE_DEVICES from Secrets: {cuda_dev}")
    except Exception:
        pass

except ImportError:
    # Not running on Kaggle or library not found
    pass

# --- IMPORTS ---
import torch
import numpy as np
import random
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

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train Swin Transformer for Fish Feeding Intensity")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/swin_tiny", help="Directory to save checkpoints")
    
    return parser.parse_args()

def main():
    # --- 1. Configuration ---
    args = parse_args()
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    SEED = args.seed
    SAVE_DIR = args.save_dir
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training Configuration:")
    print(f"- Device: {DEVICE}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Epochs: {NUM_EPOCHS}")
    print(f"- Learning Rate: {LEARNING_RATE}")
    print(f"- Seed: {SEED}")
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
