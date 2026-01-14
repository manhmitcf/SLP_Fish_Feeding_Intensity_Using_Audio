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

# Add current directory to sys.path to allow importing local modules when running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from preprocessing.base_preprocessor import BasePreprocessor
    from preprocessing import config_preprocess as config
except ImportError:
    from base_preprocessor import BasePreprocessor
    import config_preprocess as config

class ProcessedDataset(Dataset):
    """Dataset class for loading processed .pt files"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(self.data_dir, '*.pt'))
        if len(self.files) == 0:
            print(f"Warning: No .pt files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        item = torch.load(file_path)
        return {
            'audio_name': os.path.basename(file_path),
            'waveform': item['data'],
            'target': torch.tensor(item['label'], dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function to pad waveforms to the same length"""
    wav_name = [data['audio_name'] for data in batch]
    targets = torch.stack([data['target'] for data in batch])
    waveforms = [data['waveform'] for data in batch]
    
    # Find max time length in the batch
    max_time = max([w.shape[2] for w in waveforms])
    
    padded_waveforms = []
    for w in waveforms:
        pad_amount = max_time - w.shape[2]
        if pad_amount > 0:
            w = F.pad(w, (0, pad_amount))
        padded_waveforms.append(w)
        
    padded_waveforms = torch.stack(padded_waveforms)
    
    return {'audio_name': wav_name, 'waveform': padded_waveforms, 'target': targets}

class AudioPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__(config.RAW_DATA_ROOT, config.PROCESSED_DATA_ROOT)
        self.seed = config.SEED
        self.test_samples = config.TEST_SAMPLES_PER_CLASS

    def process_file(self, file_path):
        """
        Read wav file, convert to LOG-MEL spectrogram (natural log), return Tensor
        Suitable for HTS-AT / AST
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

            melspec = np.maximum(melspec, 1e-10)
            log_mel = np.log(melspec)

            tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

            return tensor

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def get_file_list(self):
        """Get list of files and labels from directory structure"""
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
        self.create_dirs()
        all_data = self.get_file_list()
        if not all_data:
            print("No files found! Check RAW_DATA_ROOT in preprocessing/config_preprocess.py")
            return

        class_data = {}
        for f, label in all_data:
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(f)

        random_state = np.random.RandomState(self.seed)
        
        for label, files in class_data.items():
            random_state.shuffle(files)
            n_test = self.test_samples
            
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
        Create DataLoaders for train, val, and test sets from processed data.
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    # Uncomment the line below to run preprocessing
    preprocessor.split_and_save()
    
    # Example of getting dataloaders
    # train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=32)
    # print("DataLoaders created successfully.")
