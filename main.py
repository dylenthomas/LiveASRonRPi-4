from tcn import TCN
from dataloader import whisperDataLoader
import sys
import os
import pickle 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.data import Dataset
from torch import optim
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Trainer:
    """Train a provided model"""
    def __init__(self,
                 model_name:str,
                 device:str,
                 epochs:int,
                 patience:int,
                 loss_fn:nn.Module,
                 optimizer:optim,
                 model,
                 dataset:Dataset,
                 num_trials_per_input:int,
                 ):
        """_summary_

        Args:
            model_name (str): Name of the model for saving to a file
            device (str): The device being trained on
            epochs (int): The number of epochs to train for
            patience (int): The patience of the model to check for overfitting (in number of epochs)
            loss_fn (nn.Module): The loss function for training
            optimizer (nn.Module): The optimizer for training
            model (nn.Module): TCN model to be trained
            dataset (Dataset): The dataloader for all data
            num_trials_per_input (int): The desired number of trials to be loaded per input
        """
        self.model_name = model_name
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.dataset = dataset
        self.num_trials_per_input = num_trials_per_input
        
        self.figure_path = os.path.join(os.path.dirname(__file__), 'figures')
        self.model_path = os.path.join(os.path.dirname(__file__), 'models')

    def train(self):
        """
        Main training function
        Returns average validation loss over training period
        """

        val_loss_list = []
        train_loss_list = []
        completed_epochs = 0
        for epoch in range(self.epochs):
            print('=' * 100)
            print(f'EPOCH { epoch + 1 }/{self.epochs}')
            print(f'MODEL NAME: {self.model_name}')

            train_loss = self.train_one_epoch()
            completed_epochs += 1

            if len(self.dataset.validation_paths) > 0:
                validation_loss = self.validate()
                val_loss_list.append(validation_loss)

                print(f'TRAIN LOSS: {train_loss}')
                print(f'VALIDATION LOSS: {validation_loss}')
                print(f'EARLY STOP COUNT: {self.counter}')

                if self.early_stop(validation_loss):
                    print('EARLY STOP CRITERIA REACHED')
                    break

            else:
                print(f'TRAIN LOSS: {train_loss}')
                train_loss_list.append(train_loss)
                
        #if len(self.dataset.test_paths) > 0:
        #    self.test()

        return sum(val_loss_list) / len(val_loss_list) if len(val_loss_list) > 0 else None, epoch + 1

    def test(self):
        """Test the model after training is complete"""
        
        self.model.eval()
        for t in range(len(self.dataset.test_paths)):
            inputs, targets = self.dataset[('Ts', t)]

            with torch.no_grad():
                pred = self.model(inputs)
                
            pred = pred.cpu()
            targets = targets.cpu()
            
            rmse = self.rmse(pred, targets)
            r2 = self.r2(pred, targets)
            
            self.plotMoments(pred, targets, rmse, r2, self.dataset.test_paths[t])
            
        self.model.train()
        self.save_model()
             
    def graph_loss(self, loss_list, completed_epochs):
        """Create a matplotlib graph of the loss per epoch and save it"""

        fig, ax = plt.subplots()
        ax.plot(range(completed_epochs), loss_list)
        ax.set_title(f'{self.model_name}')
        ax.set_xlabel('EPOCHS')
        ax.set_ylabel('LOSS (MSE)')
        
        file_name = os.path.join(self.figure_path, f'{self.model_name}.png') 
        fig.savefig(file_name, format='png')

    def train_one_epoch(self):
        """Train the model over one epoch"""
        
        #create list for the number of trials loaded at once per input
        if self.num_trials_per_input > 0:
            total_inputs = len(self.dataset.train_paths)
            num_inputs = len(self.dataset.train_paths) // self.num_trials_per_input
            remainder_trials = total_inputs % self.num_trials_per_input #account for any remainders
       
        elif self.num_trials_per_input == -1:
            num_inputs = 1
            self.num_trials_per_input = len(self.dataset.train_paths)
            remainder_trials = 0
             
        trials = [[x + (self.num_trials_per_input*_) for x in range(self.num_trials_per_input)] for _ in range(num_inputs)]
        if remainder_trials > 0: trials.append([x + (3*len(trials)) for x in range(remainder_trials)])
        
        x = 1
        for trial in trials:
            #print("Trial: {}/{}".format(x, len(trials)))
            inputs, targets = self.dataset[('Tr', trial)]
            
            pred = self.model(inputs)
            loss = self.loss_fn(pred, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            x += 1
            
        return loss.item()
   
    def validate(self):
        """Validate the model on novel subject data"""
        
        self.model.eval()
        for t in range(len(self.dataset.validation_paths)):
            inputs, targets = self.dataset[('V', t)]
            """
            for i in range(inputs.shape[0]):
                input_ = inputs[i, :, :, :]
                target = targets[i, :, :]
                
                with torch.no_grad():
                    pred = self.model(input_)
                    loss = self.loss_fn(pred, target)
            """
            with torch.no_grad():
                pred = self.model(inputs)
                loss = self.loss_fn(pred, targets)
                     
        self.model.train()
        return loss.item()

    def early_stop(self, validation_loss):
        """If validation loss doesnt improve during patience period stop training to prevent overfitting"""

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_model(self):
        for layer in self.model.layer_list:
            if type(layer) != nn.modules.container.Sequential:
                remove_parametrizations(layer.conv1, 'weight', True)
                remove_parametrizations(layer.conv2, 'weight', True)

        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, os.path.join(self.model_path, f'{self.model_name}.pt'))
        print(f'Model "{self.model_name}" successfully saved.')

    def rmse(self, preds, targets):
        return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    def r2(self, preds, targets):
        ssRes = torch.sum((targets - preds) ** 2)
        ssTot = torch.sum((targets - targets.mean()) ** 2)

        return (1 - (ssRes / ssTot)).item()

    def plotMoments(self, preds, targets, rmse, r2, trial_name):        
        fig, ax = plt.subplots()
        
        len_preds = preds.shape[-1]
        len_targets = targets.shape[-1]

        if len_preds != len_targets:
            raise ValueError('Number of elements in predictions and targets are not equal')

        idxs = torch.arange(0, len_preds)

        for i in range(preds.shape[1]): #data should be in the shape of [1, channels, length]
            label_pred = f"Predictions[R]" if i == 0 else f"Predictions[L]"
            label_targets = f"Targets[R]" if i == 0 else f"Targets[L]"
            
            ax.plot(idxs, preds[0, i, :], label=label_pred)
            ax.plot(idxs, targets[0, i, :], label=label_targets)
        
        trial_name_split = trial_name.split('\\' if sys.platform == 'win32' else '/')
        patient_trial = trial_name_split[len(trial_name_split) - 2] + trial_name_split[len(trial_name_split) - 1]

        ax.set_xlabel(f'Index (RMSE: {round(rmse, 5)}, R²: {round(r2, 5)})')
        ax.set_ylabel('Moment')
        ax.set_title(f'Predicted vs Actual Moments\n(M-{self.model_name}_[{patient_trial}])')
        ax.legend()

        file_name = os.path.join(self.figure_path, f"M-{self.model_name}_[{patient_trial}]")
        
        fig.savefig(f"{file_name}.png", format='png')
        #Save the figure as a pickle file for interactability 
        pickle.dump(fig, open(f"{file_name}.fig.pickle", 'wb'))

def get_device():
    "Return the device type for training"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Using device:[{device}]")
    return device

###VARS
NUM_FILTERS = 5
LAYERS = {'res1': (257, 512),
          'res2': (512, 256),
          'res3': (256, 128),
          'res4': (128, 64),
          'res5': (64, 32),
          'fcl1': (32, 1)
          }
DROPOUT_PROB = 0.15
CHANNELS_PER_HIDDEN_LAYER = 80
RES_MAPPING = True
WEIGHT_INIT_NAME = 'random'

LOSS_FN = "MSELoss"
OPTIMIZER = "Adam"
LR = 5e-5

MODEL_NAME = "whisper"
DEVICE = get_device()
EPOCHS = 50
PATIENCE = 25
DATALOADER = whisperDataLoader(root_dir='/Volumes/EXTREME SSD/LibriSpeechWAV',#"/storage/home/hcoda1/7/dliverman3/p-ayoung63-0/whisper_pjct/LibriSpeechWAV/",
                               dict_pth="/Users/dylenthomas/Documents/whisper/dictionary/words.txt",#"/storage/home/hcoda1/7/dliverman3/p-ayoung63-0/whisper_pjct/whisper/dictionary/words.txt",
                               sample_rate=16000,
                               device=DEVICE
                               )
NUM_TRIALS_PER_INPUT = 5

TEST_DATA = random.sample(DATALOADER.input_paths, 24000)
DATALOADER.partition_data(test_trials=TEST_DATA, validation_trials=None)

#print("Test: {}".format(len(DATALOADER.test_paths)))
#print("Train: {}".format(len(DATALOADER.train_paths)))
#print("Validation: {}".format(len(DATALOADER.validation_paths)))

VALIDATION_DATA = random.sample(list(DATALOADER.train_paths), len(DATALOADER.train_paths)//2)
DATALOADER.partition_data(test_trials=TEST_DATA, validation_trials=VALIDATION_DATA)

#print("Test: {}".format(len(DATALOADER.test_paths)))
#print("Train: {}".format(len(DATALOADER.train_paths)))
#print("Validation: {}".format(len(DATALOADER.validation_paths)))
###

if __name__ == "__main__":
    model = TCN(
        num_filters=NUM_FILTERS,
        layers=LAYERS,
        dropout_prob=DROPOUT_PROB,
        channels_per_hidden_layer=CHANNELS_PER_HIDDEN_LAYER,
        res_mapping=RES_MAPPING,
        weight_init_name=WEIGHT_INIT_NAME
    ).to(device=DEVICE)
    
    loss_fn = getattr(nn, LOSS_FN)()
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
    
    trainer = Trainer(
        model_name=MODEL_NAME,
        device=DEVICE,
        epochs=EPOCHS,
        patience=PATIENCE,
        loss_fn=loss_fn,
        optimizer=optimizer,
        model=model,
        dataset=DATALOADER,
        num_trials_per_input=NUM_TRIALS_PER_INPUT
    )
    
    #DATALOADER.partition_data(test_trials=TEST_DATA, validation_trials=VALIDATION_DATA)
    avg_val_loss = trainer.train()
    print("Average Validation Loss: {}".format(avg_val_loss))
    
    trainer.test()