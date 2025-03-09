import logging
import sys
import json
import os
import pickle 

import torch
import torch.nn as nn
from TCN.tcn import TCN
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import strokeDataset 
from torch.nn.utils.parametrize import remove_parametrizations

from data_vars import RECEPTIVE_FIELD, MOMENT_OFFSET

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Trainer:
    """Train a provided model"""
    def __init__(self,
                 model_name:str,
                 device:str,
                 epochs:int,
                 patience:int,
                 loss_fn,
                 optimizer,
                 model,
                 dataset,
                 num_trials_per_input,
                 ):

        """
            model_name(str): the name of the file to save the model to when training is complete
            device(str): the device the model is being trained on
            epochs(int): how many iterations the model is trained
            batch_size(int): the amount of data in one batch for training the model
            patience(int): how long the early stopper waits to stop the model
            loss_fn: loss function for training
            optmizer: optimizer function for training
            model: model to be trained
            train_dataset: dataset the model is to be trained on
            validation_dataset: dataset that is used to validate model preformance for hyper-parameter tuning
            test_dataset: dataset that is used to test model preformance on novel data
            test_trials: a list of trial names being used for testing
        """

        #set up logger for outputs
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(name)s: %(message)s')
        ###
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
            logging.info(f'EPOCH { epoch + 1 }/{self.epochs}')
            logging.info(f'MODEL NAME: {self.model_name}')

            train_loss = self.train_one_epoch()
            completed_epochs += 1

            if len(self.dataset.validation_paths) > 0:
                validation_loss = self.validate()
                val_loss_list.append(validation_loss)

                print(f'TRAIN LOSS: {train_loss}')
                print(f'VALIDATION LOSS: {validation_loss}')
                print(f'EARLY STOP COUNT: {self.counter}')

                if self.early_stop(validation_loss):
                    logging.info('EARLY STOP CRITERIA REACHED')
                    break

            else:
                print(f'TRAIN LOSS: {train_loss}')
                train_loss_list.append(train_loss)
                
        if len(self.dataset.test_paths) > 0:
            self.test()

        return sum(val_loss_list) / len(val_loss_list) if len(val_loss_list) > 0 else None, epoch + 1

    def test(self):
        """Test the model after training is complete"""
        
        self.model.eval()
        for t in tqdm(range(len(self.dataset.test_paths)), desc='Testing'):
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
        
        for trial in tqdm(trials, desc='Training'):
            inputs, targets = self.dataset[('Tr', trial)]
            
            pred = self.model(inputs)
            loss = self.loss_fn(pred, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
        return loss.item()
   
    def validate(self):
        """Validate the model on novel subject data"""
        
        self.model.eval()
        for t in tqdm(range(len(self.dataset.validation_paths)), desc='Validating'):
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
        logging.info(f'Model "{self.model_name}" successfully saved.')

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

class SubjectIndependenceTrainer:
    
    """Create seperate models and trainers for each subject, with MSE Loss function and Adam optimizer"""
    def __init__(self,
                 device:str,
                 epochs:int,
                 batch_size:int,
                 patience:int,
                 lr:float,
                 num_filters:int,
                 layers:dict,
                 dropout_prob:float,
                 channels_per_hidden_layer:int,
                 sensor_list:list,
                 res_mapping:bool,
                 dataset_root,
                 weight_init_name=None,
                 num_trials_per_input=1
                 ):

        """
            dataset: dataset class containing all the data
            subjects(list): should be a list of all the subjects in the training dataset
            device(str): device to train the model on
            epochs(int): number of epochs to train the model through
            batch_size(int): amount of data in a single batch for training final model
            patience(int): how long early stopping waits before stopping the model
            lr(float): learning rate for the optimizer

            *The rest of the variables are described in the TCN class*
        """

        self.sensor_list = sensor_list[1]
        self.input_channels = len(self.sensor_list)
        self.sensor_id = sensor_list[0]
        self.device = device
        print("Using: ", self.device)
        
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        
        self.num_filters=num_filters
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.channels_per_hidden_layer = channels_per_hidden_layer
        self.res_mapping = res_mapping
        self.num_trials_per_input = num_trials_per_input
        self.weight_init_name = weight_init_name
        
        self.dataset = strokeDataset(root_dir=dataset_root,
                                     batch_size=batch_size,
                                     receptive_field=RECEPTIVE_FIELD,
                                     target_offset=MOMENT_OFFSET, 
                                     input_cols=self.sensor_list,
                                     target_cols=['hip_flexion_r_moment', 'hip_flexion_l_moment'],
                                     device=self.device
                                )
        assert len(self.dataset) > 0, "No data found, likely incorrect file path"
        
        self.loss_fn = nn.MSELoss()
        self.layers['res1'] = (len(self.sensor_list), self.layers['res1'][1])

    def train(self):
        """Train all models"""

        avg_val_loss_dict = {}
        avg_epochs = []
        participants = self.dataset.persons
        for person in participants:
            model_name = person + '_' + self.sensor_id 

            self.dataset.partition_data(test_trials=['normal_walk'], validation_trials=[person])
            
            model = TCN(num_filters=self.num_filters,
                        layers=self.layers,
                        dropout_prob=self.dropout_prob,
                        channels_per_hidden_layer=self.channels_per_hidden_layer,
                        res_mapping=self.res_mapping,
                        weight_init_name=self.weight_init_name if self.weight_init_name != None else 'xavier_normal'
                        ).to(device=self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            trainer = Trainer(model_name=model_name,
                              device=self.device,
                              epochs=self.epochs,
                              patience=self.patience,
                              loss_fn=self.loss_fn,
                              optimizer=optimizer,
                              model=model,
                              dataset=self.dataset,
                              num_trials_per_input=self.num_trials_per_input
                              )

            avg_val_loss_dict[person], epochs = trainer.train()
            avg_epochs.append(epochs)
            
            #delete objects to free memory
            del model 
            exit() #stop after training one model for debugging purposes 
        print('='*100)
        print(f'Average Loss by Model: {avg_val_loss_dict}')
        self.write_avg_json(avg_val_loss_dict)
        avg_epochs = sum(avg_epochs) // len(avg_epochs)

        #Allocate all data for training
        self.dataset.partition_data(test_trials=None, validation_trials=None)

        #Train single model on all data and the number of epochs being the average from cross validation training
        final_model = TCN(num_filters=self.num_filters,
                          layers=self.layers,
                          dropout_prob=self.dropout_prob,
                          channels_per_hidden_layer=self.channels_per_hidden_layer,
                          res_mapping=self.res_mapping,
                          weight_init_name=self.weight_init_name if self.weight_init_name != None else 'xavier_normal'
                          ).to(device=self.device)

        model_name = 'full_subject_model' + '_' + self.sensor_id
        
        trainer = Trainer(model_name=model_name,
                          device=self.device,
                          epochs=avg_epochs,
                          patience=self.patience,
                          loss_fn=self.loss_fn,
                          optimizer=optimizer,
                          model=final_model,
                          dataset=self.dataset,
                          num_trials_per_input=self.num_trials_per_input
                          )
        trainer.train()

        print('='*50)
        print(f'Finished run for {self.sensor_id}')

    def write_avg_json(self, avg_val_loss_dict):
        """Write the average loss per model to a json file"""

        obj = json.dumps(avg_val_loss_dict, indent=4)
        with open('./figures/average_subj_indep_loss.json', 'w') as file:
            file.write(obj)