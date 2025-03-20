import os
import sys
import warnings
import numpy as np
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from utils.TextGrid_utils import makeArrayFromTextGrid

#https://codeocean.com/capsule/5421243/tree/v2

class whisperDataLoader(Dataset):
    def __init__(self,
                 root_dir:str,
                 dict_pth:str,
                 sample_rate:int = 441000,
                 device:str='cpu',
                 ):
        """DataLoader for speeceh to text model TCN

        Args:
            root_dir (str): parent directory for all subjects
            dict_pth (str): path to the dictionary txt file
            sample_rate (int): the sample rate to force onto every file
            device (str, optional): name of the device the model is being trained on. Defaults to 'cpu'.
        """
        
        self.root_dir = root_dir
        self.device = device
        self.data_dirs= list(os.listdir(self.root_dir)) #ground truth and input directories
        
        self.input_paths, self.ground_truth_paths = self.get_all_data_paths_()
        
        self.enforced_sample_rate = sample_rate
        self.tg_processor = makeArrayFromTextGrid(self.enforced_sample_rate, dict_pth)

        self.test_paths = []
        self.validation_paths = []
       
    def __len__(self):
        """Returns how many trials the dataset found
            for length of test, trial, and validation dataset call the length of each path list
        """
        return len(self.ground_truth_paths)
    
    def __getitem__(self, indx_tuple:Tuple[str, List[int]]):
        """Load data from a list of indecies or just index relative to trial_paths
        
        indx must contain data type indicator:
        Tr - train
        Ts - test
        V - validation
        
        dataset must be indexed as such:
            dataset[('Tr', 0)]
                or
            dataset[('Tr', [0, 1, 2, 3, 4])]
        to access the training data
        """
        dset_type = indx_tuple[0]
        indx = indx_tuple[1]
       
        match dset_type:
            case 'Tr':
                current_dataset = self.train_paths
            case 'Ts':
                current_dataset = self.test_paths
            case 'V':
                current_dataset = self.validation_paths
            case _:
                raise(AssertionError, "Must pass Tr, Ts, or V")
  
        if type(indx) == list:
            trials = [current_dataset[i] for i in indx]
        elif type(indx) == int:
            trials = [current_dataset[indx]] 
        else:
            raise ValueError(f"indx must be either List[int] or int, recieved {type(indx)}")
 
        #Create a list for each trial and store them in one list
        data = [list(self.load_trial_data(trial_path)) for trial_path in trials]

        #Zero pad tensors to allow proper concatenation if list of indxs where used
        if type(indx) == list:
            data = self.zero_pad(data)
 
        input_data, target_data = zip(*data)
        input_data = torch.cat(input_data, dim=0)
        target_data = torch.cat(target_data, dim=0)
        
        #return self.collate(input_data, target_data) 
        return input_data, target_data

    def partition_data(self, test_trials:List[str], validation_trials:List[str]):
        """Parition all of the data found in get_trial_paths into train, validation, and test sets
            Create self.validation_paths, self.test_paths, and self.train_paths
            
            test_trials (List[str]): list of trials to reserve for the test dataset. Defaults to None (no test dataset)
            validation_trials (List[str]): list of trials to reserve for the validation dataset. Defaults to None (no validation dataset)
        """
        #use sets for easy duplicate comparison between the test and validation sets 
        self.test_paths = set(self.test_paths) 
        self.validation_paths = set(self.validation_paths)
      
        #allocate paths to test dataset
        if test_trials is not None:
            assert type(test_trials) == list, "If test_trials is not None, must be a list"
            
            for path in self.input_paths:
                for req_path in test_trials:
                    if req_path in path: self.test_paths.add(path)
        
        #allocate paths to validation dataset 
        if validation_trials is not None:
            assert type(validation_trials) == list, "If validation_trials is not None, must be a list"
            
            for path in self.input_paths:
                for req_path in validation_trials:
                    if req_path in path: self.validation_paths.add(path)
        
        #check for duplicate paths in validation and test dataset
        for path in self.test_paths:
            if path in self.validation_paths: 
                warnings.warn(f'Duplicate trials exist in test and validation sets: {path}')
    
        self.train_paths = self.input_paths.copy() #create input paths
        paths_to_remove = self.test_paths | self.validation_paths #join the two sets together
        for path in paths_to_remove:
            self.train_paths.remove(path)
        
        #convert to lists so they can be used    
        self.test_paths = list(self.test_paths)
        self.validation_paths = list(self.validation_paths)
        
        self.train_paths = np.array(self.train_paths)
        self.test_paths = np.array(self.test_paths)
        self.validation_paths = np.array(self.validation_paths)
        
        #Shuffle the lists
        np.random.shuffle(self.train_paths)
        np.random.shuffle(self.test_paths)
        np.random.shuffle(self.validation_paths)
        
        #print('----After Patition----')
        #print(f'len of input paths: {len(self.train_paths)}')
        #print(f'len of test paths: {len(self.test_paths)}')
        #print(f'len of validation paths: {len(self.validation_paths)}')
 
    def zero_pad(self, data):
        """Zero pad data for proper concatenation
            data is a list with each element being a list of [input, target]"""
        trial_lens = [trial[0].shape[-1] for trial in data]
        longest = max(trial_lens)
        
        for i, trial in enumerate(data):
            if trial_lens[i] < longest:
                pad_len = longest - trial_lens[i]
                for j in range(len(trial)):
                    data[i][j] = torch.cat((trial[j], torch.zeros(1, trial[j].shape[1], pad_len, device=self.device)), dim=2)
  
        return data
    
    def load_trial_data(self, input_path):
        """Load the input and ground truth data into torch tensors

        Returns:
            Torch.Tensor: Input and Ground Truth tensors
        """
        input_tensor, sample_rate = torchaudio.load(input_path)
        input_tensor = input_tensor.to(self.device)
        #Ensure that every audio file follows the same sample rate
        if self.enforced_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.enforced_sample_rate).to(self.device)
            input_tensor = resampler(input_tensor)
            del(resampler)
        
        gt_path = self.get_gt_path_from_input_path_(input_path)
        gt_tensor = torch.Tensor(self.tg_processor.processTextGrid(gt_path, input_tensor.shape[-1]))
        gt_tensor = gt_tensor.to(device=self.device)
        
        #Normalize the data
        input_tensor = (input_tensor - torch.mean(input_tensor)) / torch.std(input_tensor)
        
        return input_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)
    
    def get_gt_path_from_input_path_(self, input_path):
        input_path = input_path.split('\\' if sys.platform == 'win32' else '/')
        input_path = input_path[len(input_path) - 1]
        trial_name = input_path.replace('.wav', '')
        
        for path in self.ground_truth_paths:
            if trial_name in path:
                gt_path = path
                
        return gt_path
    
    def get_all_data_paths_(self):
        """Get a list of all desired inputs and ground truths"""
        input_paths = []
        ground_truth_paths = []
        
        for dir_ in self.data_dirs:
            dir_ = os.path.join(self.root_dir, dir_)
            
            for dirpath, _, filenames in os.walk(dir_):
                for filename in filenames:
                    if filename.endswith('.TextGrid'): #ground truth
                        ground_truth_paths.append(os.path.join(dirpath, filename))
                    elif filename.endswith('.wav'): #input
                        input_paths.append(os.path.join(dirpath, filename))
            
        return input_paths, ground_truth_paths
    
if __name__ == '__main__':
    dl = whisperDataLoader("/Volumes/EXTREME SSD/LibriSpeechWAV", 44100)
    dl.partition_data(None, None)
    
    indxs = [_ for _ in range(len(dl.input_paths))]
    chunk_size = 1000
    chunks = len(indxs) // chunk_size
    remainder = len(indxs) % chunk_size
    
    last = 0
    for i in range(chunks):
        chunk = indxs[last:last+chunk_size]
        last += chunk_size
    
        print(dl[('Tr', chunk)])
    
    chunk = indxs[last:last+remainder]
    print(dl[('Tr', chunk)])
    