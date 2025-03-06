DATA_DIR = "/Volumes/EXTREME SSD/LibriSpeech"
GT_PATH = "ground_truths"

from torch.utils.data import Dataset
import os

class LibriSpeechDataloader(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        """
        Dataloader for training a Speech To Text model (STT) using the LibriSpeech Dataset
        """
        
        self.gt_path = GT_PATH
        
        self.trans_paths = self.collect_trans(dataset_dir)
        all_trials = list(self.trans_paths.keys())
        print(f'{len(all_trials)} files found.')
        
        with open("/Volumes/EXTREME SSD/LibriSpeech/train-clean-100/19/227/19-227.trans.txt", 'r') as file:
            txt = file.read()
            print(txt.split('\n')[0])
            print(txt.split('\n')[1])
            
        #Create ground truths in necessary
        gt_dirs = next(os.walk(self.gt_path))[1]
        
        for t in all_trials:
            if t not in gt_dirs:
                self.make_gts(t)

    def make_gts(self, trial):
        """
            Create ground truth csvs
        """
        #Get transcript data
        with open(self.trans_paths[trial], 'r') as f:
            trans = f.read().split('\n')
        f.close()
        
        #Create directory to store ground truth data
        trial_name = trial.replace('.trans.txt', '')
        os.mkdir(os.path.join(self.gt_path, trial_name))
        
        
        




    def collect_trans(self, dataset_dir):
        """
            Get all the paths for the transcripts
            
            Returns: A dictionary of paths
        """
        dict = {}
        for dirpath, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith('.trans.txt'):
                    path = os.path.join(dirpath, filename)
                    dict[filename] = path
                
        return dict
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    LibriSpeechDataloader(DATA_DIR)