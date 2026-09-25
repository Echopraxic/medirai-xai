import numpy.typing as npt

import torch
import os
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import TensorDataset, DataLoader

#from medirai_base_model import MediraiEnsembleModelV1
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
from tqdm import tqdm


class Predictor():
    '''
    Wrapper class for prediction MLP networks that applied
    to outputs that do not match the requried number of 
    classes.

    Attributes
    ----------
    input_size : int
        length or number of components for the input layer.
    hidden_size : int
        size of the hidden layer
    output_size : int
        number of classes
    dropout_pob : float
        likely hood of dropout for training
    '''

    def __init__(self, 
                 input_size : int,
                 hidden_size : int, 
                 output_size : int = 2,
                 dropout_prob : float = 0.5):

        self.pred_net = PredictorNet(input_size, hidden_size, output_size=output_size, dropout_prob=dropout_prob)

    def train(self, x : npt.NDArray, y : npt.NDArray, savename : str, **kwargs) -> None:
        '''
        Train the MLP

        Parameters
        ----------
        x : npt.NDArray
            training data as an array
        y : npt.NDArray
            training labels as an array
        savename : str
            name to used when freezing weights during training
        '''

        trainer = PredictorNetTrainer(self.pred_net, x, y, savename, **kwargs)
        trainer.train()


    def load(self, path_to_state_dict : str, device : str) -> None:
        '''
        Load frozen weights from disc.

        Parameters
        ----------
        path_to_state_dict : str
            path to frozen pytorch state dictionary
        device : str
            map loaded netwrk to device. Must be 'cpu' or 'cuda'
        '''

        self.pred_net.load_state_dict(
                            torch.load(
                                path_to_state_dict, 
                                map_location=torch.device(device),
                                ),
                            )
        self.pred_net.eval()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Perform an inference pass of the network with the 
        provided input

        Parameters
        ----------
        x : torch.Tensor
            values to get a prediction for using the network

        Returns
        -------
        pred : torch.Tensor
            logits of predictions
        '''
        return self.pred_net(x)

    def get_hidden(self, x :  torch.Tensor) -> torch.Tensor:
        '''
        Helper function to get the hidden layer feature
        representations.

        Parameters
        ----------
        x : torch.Tensor
            input values to extract the hidden features for.

        Returns
        -------
        feat_rep : torch.Tensor
            Values of activations fuctions in the hidden layer
        '''
        return self.pred_net.get_hidden(x)



class PredictorNet(nn.Module):
    '''
    Pytroch implemenation of a three layer MLP

    Attributes
    ----------
    input_size : int
        size of the input, number of components
    hidden_size : int
        size of the hidden layer
    output_size : int
        number of classes
    dropout_prob : float
        rate to use for dropout
    '''

    def __init__(self, 
                 input_size : int, 
                 hidden_size : int, 
                 output_size : int = 2, 
                 dropout_prob : float = 0.5):
        
        super(PredictorNet, self).__init__()
        # Legacy, unused in forward(); kept only so existing checkpoints still load with strict=True.
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU6(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size),
            nn.ReLU6()
            )

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU6()
        self.dropout = nn.Dropout(dropout_prob)


    def get_hidden(self, feat_rep : torch.Tensor) -> torch.Tensor:
        '''
        Helper function to set up easy access to
        values in the hidden layer.

        Parameters
        ----------
        feat_rep : torch.Tensor
            values to feed in the network and extract
            the hidden represenations

        Returns
        -------
        x : torch.Tensor
            hidden layer values
        '''
        
        x = self.fc1(feat_rep)
        x = self.relu(x)

        return x
        

    def forward(self, feat_rep : torch.Tensor) -> torch.Tensor:
        '''
        Complete a forward pass of the network using the
        provided input.

        Parameters
        ----------
        feat_rep : torch.Tensor
            values to complete the forward pass on

        Returns
        -------
        x : torch.Tensor
            logits for classification
        ''' 
        x = self.get_hidden(feat_rep)
        x = self.dropout(x)
        x = self.fc2(x)
        # No activation on the logits (CODEBASE_TODO P0-14): a ReLU6 here clamped logits to [0, 6], so
        # whenever both were <= 0 the output tied and argmax always returned class 0.

        return x
    

class PredictorNetTrainer():
    '''
    Helper class to trainer any isntance of MLP

    Attributes
    ----------
    model : PredictorNet
        MLP network to be trained
    feat_rps : npt.Array
        Training data
    labels : npt.Array
        Training labels
    save_name : str
        Name of network to use when freezing weights
        during training
    num_epochs : int
        Number of times training will loop over data
    val_split : float
        Percentage of training data hold back for a 
        validation
    lr : float
        Learning rate of learning protocol
    weight_decay : float
        weight_dccay rate to using in training
    '''
    def __init__(
        self,
        model : PredictorNet,
        feat_reps : npt.NDArray,
        labels : npt.NDArray,
        save_name : str,
        num_epochs : int = 20,
        val_split : float = 0.1,
        lr : float = 1e-4,
        weight_decay : float = 5e-5,
        batch_size : int = 64,
        groups : npt.NDArray | None = None,
        seed : int = 42,
        ):
        '''
        groups : optional source-image id per row. Pass it whenever feat_reps contains several
            augmented copies of the same image (e.g. the DermFoundation 7x augmentation), so that
            copies never end up on both sides of the train/val split.
        '''
        self.model = model
        self.batch_size = batch_size
        self.groups = groups
        self.seed = seed
        self.feat_reps = feat_reps
        self.labels = labels
        self.save_name = save_name
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.lr = lr
        self.weight_decay = weight_decay
    
    
    def _init_data(self) -> tuple[DataLoader, DataLoader]:
        '''
        Helper function to prepar the data for training
        via pytorch

        Returns
        -------
        train_loader : DataLoader
            pytorch DataLoader for training
        val_loader :  DataLoader
            pytorch DataLoader for validation

        '''
        if self.groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=self.val_split, random_state=self.seed)
            tr, va = next(gss.split(self.feat_reps, self.labels, self.groups))
            X_train, X_val, y_train, y_val = self.feat_reps[tr], self.feat_reps[va], self.labels[tr], self.labels[va]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                        self.feat_reps,
                        self.labels,
                        test_size = self.val_split,
                        random_state=self.seed,
                        stratify=self.labels,
            )
        
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        
        # was DataLoader(dataset): batch size 1 and no shuffling, i.e. samples in CSV (class-sorted) order (P0-16)
        generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader


    def train(self) -> None:
        '''
        Train the MLP nework using predetermined
        training protocol
        '''

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
                            self.model.parameters(), 
                            lr = self.lr,
                            weight_decay = self.weight_decay,
                        )
            
        iter = 0
        
        train_loader, val_loader = self._init_data()
        
        best_val_acc = -1.0
        for epoch in range(self.num_epochs):
            print(f'epoch {epoch+1}/{self.num_epochs}')

            self.model.train()
            correct = 0
            total = 0
            for i, (feat_rep, labels) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                outputs = self.model(feat_rep)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()
                iter = iter + 1

            accuracy = 100 * correct / total
            print(f'Train accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}')

            # validation in eval mode (dropout off) and without gradients (P0-16)
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for feat_rep, labels in val_loader:
                    outputs = self.model(feat_rep)
                    _, predicted = torch.max(outputs, 1)
                    total = total + labels.size(0)
                    correct = correct + (predicted == labels).sum()

            val_accuracy = float(100 * correct / total)
            print(f'Val accuracy epoch {epoch+1}: {val_accuracy:.4f}')

            torch.save(self.model.state_dict(), f'{self.save_name}_epoch-{epoch}.plk')
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), f'{self.save_name}_best.plk')
                print(f'	new best val accuracy; saved {self.save_name}_best.plk')

        self.model.eval()

            
