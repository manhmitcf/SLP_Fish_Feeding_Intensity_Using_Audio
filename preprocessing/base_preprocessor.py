from abc import ABC, abstractmethod
import os

class BasePreprocessor(ABC):
    """
    Abstract Base Class for data preprocessing pipeline:
    Raw Audio -> Spectrogram -> Save to Disk (Train/Val/Test split)
    """
    
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
    @abstractmethod
    def process_file(self, file_path):
        """Process a single audio file into a spectrogram"""
        pass

    @abstractmethod
    def split_and_save(self):
        """Split data into train/val/test and save to processed directory"""
        pass
    
    @abstractmethod
    def get_dataloaders(self, batch_size):
        """Load processed data and return DataLoaders for train, val, and test"""
        pass

    def create_dirs(self):
        """Create output directory structure"""
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.processed_data_path, split)
            os.makedirs(split_path, exist_ok=True)
            # Create sub-folders for each class if needed, or save together with labels
