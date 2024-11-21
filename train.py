import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import dataset as Dataset
from tcn import TCN
from torch.nn.utils.parametrize import remove_parametrizations

class Trainer:
    def __init__(self,
                 epochs,
                 device,
                 batch_size,
                 patience,
                 model,
                 loss_fn,
                 optimizer,
                 dataset,
                 model_save_name,
                 dataset_split
                 ):

        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG) #Setup logging

        self.epochs = epochs
        self.device = device

        #Data and model
        logging.info('initializing model..')
        self.model = model
        self.batch_size = batch_size

        logging.info('Creating Dataloaders...')
        self.train_dataloader, self.validation_dataloader = self.partition_dataset(dataset, dataset_split)
        ###

        #Training functions
        self.train_loss_fn = loss_fn
        self.validate_loss_fn = loss_fn
        self.optimizer = optimizer
        ###

        #Early stopping stuff
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')
        ###

        self.model_save_name = model_save_name

    def train(self):
        """Main training function"""
        for _ in range(self.epochs):
            print('#' * 100)
            logging.info(f'EPOCH {_ + 1}/{self.epochs}')

            train_loss = self.train_one_epoch()
            validation_loss = self.validate()
            if self.early_stop(validation_loss):
                logging.info('Early stop reached')
                break

        self.save_model()

    def train_one_epoch(self):
        """Train the model for one epoch"""
        for input_, target in self.train_dataloader:
            input_, target = input_.to(device=self.device), target.to(device=self.device)

            prediction = self.model(input_)

            loss = self.train_loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'TRAIN LOSS: {loss.item()}')
        return loss.item()

    def validate(self):
        """Validate model on novel data to test for early stopping"""
        self.model.eval()

        for input_, target in self.validation_dataloader:
            input_, target = input_.to(device=self.device), target.to(device=self.device)

            with torch.no_grad():
                prediction = self.model(input_)

                loss = self.validate_loss_fn(prediction, target)

        print(f'VALIDATION LOSS: {loss.item()}')
        print(f'EARLY STOP COUNT: {self.counter}')
        self.model.train()
        return loss.item()

    def early_stop(self, validation_loss):
        """If the validation loss stops improving after the patience is expired stop training"""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def partition_dataset(self, dataset, dataset_split):
        """Split dataset into train and validation dataloaders"""
        indicies = torch.randperm(len(dataset)).tolist()
        train_dataset = Subset(dataset, indicies[:-dataset_split])
        validation_dataset = Subset(dataset, indicies[-dataset_split:])

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        return train_dataloader, validation_dataloader

    def save_model(self):
        """Save the model to torch script"""
        self.model.to(device='cpu')

        self.remove_model_params()

        jit_model = torch.jit.script(self.model)
        jit_model.save(self.model_save_name)
        logging.info(f'Model saved to {self.model_save_name}')

    def remove_model_params(self):
        """Remove the weight norm parametrizations on the model so it can be used in torch script"""
        for layer in self.model.layer_list:
            if type(layer) != torch.nn.modules.container.Sequential: #Doesn't have params
                remove_parametrizations(layer.conv1, tensor_name='weight', leave_parametrized=True)
                remove_parametrizations(layer.conv2, tensor_name='weight', leave_parametrized=True)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    layers = {'res1': (64, 128),
              'res2': (128, 256),
              'res3': (256, 512),
              'res4': (512, 256),
              'fcl1': (91 * 256, 2)
              }
    model = TCN(num_filters=4,
                layers=layers,
                dropout_prob=0.3,
                channels_per_hidden_layer=20,
                res_mapping=True
                ).to(device=device)

    print(model.parameters())
    print(f'USING DEVICE: {device}')

    #loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    dataset = Dataset(root_path='./full_dataset',
                      sample_rate=44100,
                      num_samples=44100,
                      device=device
                      )

    model_trainer = Trainer(epochs=200,
                            device=device,
                            batch_size=64,
                            patience=25,
                            model=model,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            dataset=dataset,
                            model_save_name='whisper.pt',
                            dataset_split=1500
                            )

    #model_trainer.train()
