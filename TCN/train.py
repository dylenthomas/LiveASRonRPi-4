import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from dataset import dataset as Dataset
from tcn import TCN
from torch.nn.utils.parametrize import remove_parametrizations

writer = SummaryWriter('runs/TCN_exp_1')

class Trainer:
    def __init__(self,
                 epochs:int,
                 device:str,
                 patience:int,
                 model_save_name:str,
                 model,
                 loss_fn,
                 optimizer,
                 train_dataloader,
                 validation_dataloader,
                 do_plot_model:bool = True,
                 fold:int = -1
                 ):

        """
            Train and validate a given model on given data
            
            epochs(int): number of epochs to train a model
            device(str): the device to train on
            patience(int): number of epochs the early stopper waits to stop, to disable early stop set patience to -1
            model_save_name(str): name of the model to be saved to, omitting the file type
            model: model to be trained
            loss_fn: loss function
            optimizer: optimizer
            train_dataloader: dataloader containing training data
            validation_dataloader: dataloader containing validation data 
            do_plot_model(bool)[Optional]: whether or not the model should be plotted in tensorboard
            fold(int)[Optional]: when doing K-Fold cross validation pass the current fold to fold, if not leave at -1
             
            Dictionary containing training results
        """

        self.epochs = epochs
        self.device = device
        self.model = model
        self.patience = patience
        self.train_loss_fn = loss_fn
        self.validate_loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.fold = '_' + str(fold) if fold > 0 else ''
        self.model_save_name = model_save_name + self.fold + '.pt'
         
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.results = {
            'Training Loss': [],
            'Validation Loss': [] 
        }
        
        if do_plot_model: 
            self.plot_model()

    def train(self):
        """Main training function"""
        for e in range(self.epochs):
            print('#' * 100)
            print(f'EPOCH: {e + 1}')
                
            print('Training started') 
            train_loss = self.train_one_epoch(self.train_dataloader)
            print('Training complete, validation started.')
            validation_loss = self.validate(self.validation_dataloader)
            print('Validation complete.')
                
            if self.early_stop(validation_loss) and self.patience >= 0:
                break
               
            self.results['Training Loss'].append(train_loss)
            self.results['Validation Loss'].append(validation_loss)
            self.plot_scalar('Training Loss' + self.fold, train_loss, e)
            self.plot_scalar('Validation Loss' + self.fold, validation_loss, e) 
                
        self.save_model()
        return self.results

    def train_one_epoch(self, train_dataloader):
        """Train the model for one epoch"""
       
        for input_, target in train_dataloader:
            input_, target = input_.to(device=self.device), target.to(device=self.device)
            prediction = self.model(input_)
           
            loss = self.train_loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def validate(self, validation_dataloder):
        """Validate model on novel data to test for early stopping"""
        self.model.eval()

        for input_, target in validation_dataloder:
            input_, target = input_.to(device=self.device), target.to(device=self.device)

            with torch.no_grad():
                prediction = self.model(input_)
                loss = self.validate_loss_fn(prediction, target)

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

    def save_model(self):
        """Save the model to torch script"""
        self.model.to(device='cpu')

        self.remove_model_params()

        jit_model = torch.jit.script(self.model)
        jit_model.save(self.model_save_name)
       
        print(f'Model saved to {self.model_save_name}')

    def remove_model_params(self):
        """Remove the weight norm parametrizations on the model so it can be used in torch script"""
        for layer in self.model.layer_list:
            if type(layer) != torch.nn.modules.container.Sequential: #Doesn't have params
                remove_parametrizations(layer.conv1, tensor_name='weight', leave_parametrized=True)
                remove_parametrizations(layer.conv2, tensor_name='weight', leave_parametrized=True)
               
    def plot_model(self):
        """Send graph of the model to tensorboard"""
        writer.add_graph(self.model, next(iter(self.train_dataloader))[0]) 
        writer.close()
        
    def plot_scalar(self, scalar_name:str, scalar_value, global_step:int):
        """Send a scalar value to tensorboard to plot"""
        writer.add_scalar(scalar_name, scalar_value, global_step)
        writer.close()
        
def plot_features(dataset, n):
    """Send features from a dataset to tensorboard to be plotted and visualized"""
        
    inputs = torch.tensor([])
    targets = []
    inds = torch.randperm(len(dataset))
        
    for i in range(n):
        indx = inds[i]
        input_, target = dataset[indx]
        input_ = input_.reshape(1, 64*91)
            
        inputs = torch.cat((inputs, input_), 0)
        targets.append(target)
        
    writer.add_embedding(inputs, targets)
    writer.close()
               
def partition_dataset(dataset, K, i, batch_size):
    """Split dataset into train and validation dataloaders based on K-Fold Cross-Validation"""
       
    usable = K * (len(dataset) // K) #make sure length of dataset is compabtible with the split
    indecies = torch.arange(0, usable).unsqueeze(0).view(K, len(dataset) // K) #cast to shape (K, size of partition)
        
    train_inds = torch.tensor([], dtype=int)
    val_inds = indecies[i, :] 
        
    for x in range(K):
        if x != i:
            train_inds = torch.cat((train_inds, indecies[x, :]), dim=-1)
    
    train_dataset = Subset(dataset, train_inds.tolist())
    validation_dataset = Subset(dataset, val_inds.tolist())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader
   
def del_objects(*args):
    """Delete training objects to clear memory after each fold"""
    for obj in args:
        del(obj)
        
if __name__ == '__main__':
    k = 3
    batch_size = 64
    epochs = 1 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layers = {'res1': (64, 128),
              'res2': (128, 256),
              'res3': (256, 128),
              'res4': (128, 64),
              'fcl1': (91 * 64, 6)
              } #number of layers for 91 units of receptive field
    dataset = Dataset(root_path='./full_dataset',
                      sample_rate=44100,
                      num_samples=44100,
                      device=device
                      )
    #plot_features(dataset, 2000)

    for i in range(k):
        model = TCN(num_filters=4,
                    layers=layers,
                    dropout_prob=0.3,
                    channels_per_hidden_layer=20,
                    res_mapping=True
                    ).to(device=device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        train_dataloader, validation_dataloader = partition_dataset(dataset, k, i, batch_size)

        model_trainer = Trainer(epochs=epochs,
                                device=device,
                                patience=-1,
                                model_save_name='whisper',
                                model=model,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                train_dataloader=train_dataloader,
                                validation_dataloader=validation_dataloader,
                                fold=i + 1,
                                do_plot_model=True
                                )
        
        results = model_trainer.train()
        del_objects(model, loss_fn, optimizer, train_dataloader, validation_dataloader, model_trainer)