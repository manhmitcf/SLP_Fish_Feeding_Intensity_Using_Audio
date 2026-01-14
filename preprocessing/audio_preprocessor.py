import os
import glob
import numpy as np
import librosa
import torch
import shutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

try:
    from preprocessing.base_preprocessor import BasePreprocessor
    from preprocessing import config_preprocess as config
except ImportError:
    from base_preprocessor import BasePreprocessor
    import config_preprocess as config

class ProcessedDataset(Dataset):
    """
    Dataset class for loading processed .pt files (Log-Mel Spectrograms).
    
    This dataset loads pre-processed tensors from disk and resizes them 
    to a fixed size (224x224) using Lanczos interpolation, making them 
    ready for models like Swin Transformer.

    Attributes:
        data_dir (str): Path to the directory containing .pt files.
        files (list): List of all .pt file paths in the directory.
        resize (torchvision.transforms.Resize): Transform to resize spectrograms.
    """
    def __init__(self, data_dir):
        """
        Initializes the ProcessedDataset.

        Args:
            data_dir (str): Directory containing the processed .pt files.
        """
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(self.data_dir, '*.pt'))
        if len(self.files) == 0:
            print(f"Warning: No .pt files found in {self.data_dir}")
        
        # Initialize Resize transform with LANCZOS interpolation
        # Target size is (224, 224) to match Swin Transformer input requirements
        self.resize = T.Resize((224, 224), interpolation=InterpolationMode.LANCZOS)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'audio_name' (str): Name of the audio file.
                - 'waveform' (Tensor): Resized Log-Mel Spectrogram [1, 224, 224].
                - 'target' (Tensor): Class label (integer).
        """
        file_path = self.files[index]
        item = torch.load(file_path)
        
        waveform = item['data'] # Original Shape: [1, 128, Time]
        
        # --- RESIZE TO 224x224 USING LANCZOS ---
        # torchvision.transforms.Resize expects input Tensor [C, H, W]
        # This ensures the output is always [1, 224, 224] regardless of original time length
        waveform = self.resize(waveform)
        
        return {
            'audio_name': os.path.basename(file_path),
            'waveform': waveform,
            'target': torch.tensor(item['label'], dtype=torch.long)
        }

def collate_fn(batch):
    """
    Collate function to combine a list of samples into a batch.
    
    Since all waveforms are already resized to (224, 224) in __getitem__,
    this function simply stacks them together.

    Args:
        batch (list): List of dictionaries returned by __getitem__.

    Returns:
        dict: A dictionary containing batched data:
            - 'audio_name' (list): List of audio filenames.
            - 'waveform' (Tensor): Batched waveforms [Batch_Size, 1, 224, 224].
            - 'target' (Tensor): Batched labels [Batch_Size].
    """
    wav_name = [data['audio_name'] for data in batch]
    targets = torch.stack([data['target'] for data in batch])
    
    # Stack waveforms into a single tensor
    waveforms = torch.stack([data['waveform'] for data in batch])
    
    return {'audio_name': wav_name, 'waveform': waveforms, 'target': targets}

class AudioPreprocessor(BasePreprocessor):
    """
    Main class for audio preprocessing pipeline.
    
    Handles reading raw WAV files, converting them to Log-Mel Spectrograms,
    splitting into Train/Val/Test sets, and saving as .pt files.
    """
    def __init__(self):
        """Initializes the AudioPreprocessor with paths and config."""
        super().__init__(config.RAW_DATA_ROOT, config.PROCESSED_DATA_ROOT)
        self.seed = config.SEED
        self.test_samples = config.TEST_SAMPLES_PER_CLASS

    def process_file(self, file_path):
        """
        Reads a WAV file and converts it to a Log-Mel Spectrogram.

        Args:
            file_path (str): Path to the input .wav file.

        Returns:
            Tensor: Log-Mel Spectrogram tensor with shape [1, n_mels, time_steps].
                    Returns None if processing fails.
        """
        try:
            wav, _ = librosa.load(file_path, sr=config.SAMPLE_RATE)

            melspec = librosa.feature.melspectrogram(
                y=wav,
                sr=config.SAMPLE_RATE,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                win_length=config.WIN_LENGTH,
                n_mels=config.NUM_MEL_BINS,
                power=2.0
            )

            # Avoid log(0) by taking maximum with a small epsilon
            melspec = np.maximum(melspec, 1e-10)
            log_mel = np.log(melspec)

            # Convert to Tensor and add channel dimension: [1, n_mels, time]
            tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

            return tensor

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def get_file_list(self):
        """
        Scans the raw data directory to find all .wav files and their labels.

        Returns:
            list: A list of tuples (file_path, label_index).
        """
        all_files = []
        for class_name, label in config.CLASS_MAPPING.items():
            class_dir = os.path.join(self.raw_data_path, class_name)
            if not os.path.exists(class_dir):
                continue
            files = glob.glob(os.path.join(class_dir, '**', '*.wav'), recursive=True)
            print(f"Found {len(files)} files for class '{class_name}'")
            for f in files:
                all_files.append((f, label))
        return all_files

    def split_and_save(self):
        """
        Splits the dataset into Train, Validation, and Test sets and saves processed files.
        
        Splitting logic:
        - If enough samples: Uses fixed number of test samples (config.TEST_SAMPLES_PER_CLASS).
        - If scarce samples: Falls back to 80-10-10 ratio split.
        """
        self.create_dirs()
        all_data = self.get_file_list()
        if not all_data:
            print("No files found! Check RAW_DATA_ROOT in preprocessing/config_preprocess.py")
            return

        # Group files by label
        class_data = {}
        for f, label in all_data:
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(f)

        random_state = np.random.RandomState(self.seed)
        
        for label, files in class_data.items():
            random_state.shuffle(files)
            n_test = self.test_samples
            
            # Determine split strategy based on data size
            if len(files) < n_test * 2:
                print(f"Warning: Not enough files for class {label}. Using ratio split 80-10-10.")
                n_total = len(files)
                test_files = files[:int(0.1*n_total)]
                val_files = files[int(0.1*n_total):int(0.2*n_total)]
                train_files = files[int(0.2*n_total):]
            else:
                test_files = files[:n_test]
                val_files = files[n_test:2*n_test]
                train_files = files[2*n_test:]
            
            print(f"Class {label}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            
            self._save_subset(train_files, label, 'train')
            self._save_subset(val_files, label, 'val')
            self._save_subset(test_files, label, 'test')

    def _save_subset(self, file_list, label, split):
        """
        Helper function to process and save a subset of files.

        Args:
            file_list (list): List of file paths to process.
            label (int): Class label for these files.
            split (str): Name of the split ('train', 'val', 'test').
        """
        save_dir = os.path.join(self.processed_data_path, split)
        for i, file_path in enumerate(tqdm(file_list, desc=f"Processing {split} - Class {label}")):
            tensor = self.process_file(file_path)
            if tensor is not None:
                base_name = os.path.basename(file_path).replace('.wav', '')
                out_name = f"{label}_{base_name}_{i}.pt"
                out_path = os.path.join(save_dir, out_name)
                torch.save({'data': tensor, 'label': label}, out_path)

    def get_dataloaders(self, batch_size=32):
        """
        Creates PyTorch DataLoaders for Train, Validation, and Test sets.

        Args:
            batch_size (int): Number of samples per batch. Default is 32.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
                   Returns (None, None, None) if processed data is missing.
        """
        train_dir = os.path.join(self.processed_data_path, 'train')
        val_dir = os.path.join(self.processed_data_path, 'val')
        test_dir = os.path.join(self.processed_data_path, 'test')

        # Check if directories exist
        if not os.path.exists(train_dir):
            print("Processed data not found. Please run split_and_save() first.")
            return None, None, None

        train_dataset = ProcessedDataset(train_dir)
        val_dataset = ProcessedDataset(val_dir)
        test_dataset = ProcessedDataset(test_dir)

        # --- FIX: Check if datasets are empty ---
        if len(train_dataset) == 0:
            print("Processed data directories exist but are empty.")
            return None, None, None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    # Uncomment the line below to run preprocessing
    # preprocessor.split_and_save()
    
    # Example of getting dataloaders
    # train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=32)
    # print("DataLoaders created successfully.")
