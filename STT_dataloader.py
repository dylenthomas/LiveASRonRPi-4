DATA_DIR = "/Volumes/EXTREME SSD/LibriSpeech"

import os

from torch.utils.data import Dataset
from utils.TextGrid_utils import makeArrayFromTextGrid

class LibriSpeechDataloader(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        """
        Dataloader for training a Speech To Text model (STT) using the LibriSpeech Dataset
        """
            

if __name__ == "__main__":
    LibriSpeechDataloader(DATA_DIR)