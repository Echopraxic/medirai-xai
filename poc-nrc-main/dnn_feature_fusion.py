from re import A
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
import numpy.typing as npt
import cv2
import pandas as pd

#from trainer import Trainer
from torch import nn, optim

from isic_data_loader import ISICDataset, ISICMixupDataset, ISICFeatureFusionMixupDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision import models
from torchvision.transforms import v2

from medirai_base_model import MediraiEnsembleModelV1

class MediraiFeatureFusionEnsemble(nn.Module):
    '''
    Model Class to use feature representations of all four DNNs to make a single binary
    classification prediction. This class with take the hidden layers of each
    MLP that is placed at the end of each DNN, concatenate them, and make a prediction.

    Attributes
    ----------
    dense_net : Type Unknown
        dense net model from MediraiEnsembleModelV1
    efficient_net : Type Unknown
        effiicint net model from MediraiEnsembleModelV1
    inception : Type Unknown
        inception model from MediraiEnsembleModelV1
    res_net : Type Unknwon
        rest net model from MediraiEnsembleModelV1
    h_feat_size : int
        Size of the hidden layers to fuse within each DNN
    h_fusion_size : int
        hidden layer size, post feature fusion operation.
    nb_classes : int
        number of output classes
    device : str
        Device to map the model to. Must be 'cpu' or 'cuda'
        
    '''

    def __init__(self, 
            dense_net = None,
            efficient_net = None, 
            inception = None, 
            res_net = None, 
            h_feat_size = 256,
            h_fusion_size = 512, 
            nb_classes = 2,
            device = 'cuda'
        ):

        super(MediraiFeatureFusionEnsemble, self).__init__()

        self.h_feat_size = h_feat_size
        self.h_fusion_size = h_fusion_size

        self.dense_net = dense_net
        self.efficient_net = efficient_net
        self.inception = inception
        self.res_net = res_net

        for param in self.dense_net.parameters():
            param.requires_grad = False
        for param in self.efficient_net.parameters():
            param.requires_grad = False
        for param in self.inception.parameters():
            param.requires_grad = False
        for param in self.res_net.parameters():
            param.requires_grad= False

        self.fc1 = nn.Linear(h_feat_size*4, h_fusion_size)
        self.fc2 = nn.Linear(h_fusion_size, nb_classes)
        self.relu = nn.ReLU6()
        self.dropout = nn.Dropout(0.5)

        self._gen_all_feature_extractors()

        self.device = device


    def dense_net_feat_forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Helper function to do forward pass on feature exraction
        of Dense Net

        Parameters
        ----------
        x : pytorch.Tensor
            input image as tensor
        
        Returns
        -------
        feature_extraction : pytorch.Tensor
            extract feature representation
        '''
        
        return self.dense_net_extractor(x.clone())['classifier.1']
    
    def efficient_net_feat_forward(self, x):
        '''
        Helper function to do forward pass on feature exraction
        of Efficient Net

        Parameters
        ----------
        x : pytorch.Tensor
            input image as tensor
        
        Returns
        -------
        feature_extraction : pytorch.Tensor
            extract feature representation
        '''
        return self.efficient_net_extractor(x.clone())['classifier.1.1']

    def inception_feat_forward(self, x):
        '''
        Helper function to do forward pass on feature exraction
        of Inception 

        Parameters
        ----------
        x : pytorch.Tensor
            input image as tensor
        
        Returns
        -------
        feature_extraction : pytorch.Tensor
            extract feature representation
        '''
        return self.inception_extractor(x.clone())['fc.1']

    def res_net_feat_forward(self, x):
        '''
        Helper function to do forward pass on feature exraction
        of ResNet
 
        Parameters
        ----------
        x : pytorch.Tensor
            input image as tensor
        
        Returns
        -------
        feature_extraction : pytorch.Tensor
            extract feature representation
        '''
        return self.res_net_extractor(x.clone())['fc.1']

    def get_hidden(self, x_224 : torch.Tensor, x_299 : torch.Tensor) -> torch.Tensor:
        '''
        Get all feature representations from the hidden layers on MLPS
        that are placed on the DNNs, concatenate them.

        Parameters
        ----------
        x_224 : torch.Tensor
            input value reshaped to be (224, 224, 3)
        x_299 : torch.Tensor
            input value reshaped to be (299, 299, 3)

        Returns
        -------
        x : torch.Tensor
            concatenated hidden values
        '''

        x_dense = self.dense_net_feat_forward(x_224)
        x_efficient = self.efficient_net_feat_forward(x_224)
        x_inception = self.inception_feat_forward(x_299)
        x_res_net = self.res_net_feat_forward(x_224)

        x = torch.cat((x_dense, x_efficient, x_inception, x_res_net), dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        return x
    

    def forward(self, x_224 : torch.Tensor, x_299 : torch.Tensor) -> torch.Tensor:
        '''
        Forward operation of the Feature fusion model.

        Parameters
        ----------
        x_224 : torch.Tensor
            input value reshaped to be (224, 224, 3)
        x_299 : torch.Tensor
            input value reshaped to be (299, 299, 3)

        Returns
        -------
        x : torch.Tensor
            logits for classification task
        '''
        #x = self.classifier(x)
        x = self.get_hidden(x_224, x_299)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        return x
    
    def _gen_all_feature_extractors(self) -> None:
        '''
        Helper function to define all the feature extractions.
        '''

        self.dense_net_extractor = create_feature_extractor(
            self.dense_net,
            return_nodes=['classifier.1']
        )

        self.efficient_net_extractor = create_feature_extractor(
            self.efficient_net,
            return_nodes = ['classifier.1.1']
        )

        self.inception_extractor = create_feature_extractor(
            self.inception,
            return_nodes = ['fc.1']
        )

        self.res_net_extractor = create_feature_extractor(
            self.res_net,
            return_nodes = ['fc.1']
        )    


class FFModel:
    '''
    Wrapper clss for the DNN feature fusion model

    Attributes
    ----------
    model : MediraiFeatureFusionEnsemble
        feature fusion model applied to DNNs
    '''

    def __init__(self, model):

        self.model = model

    def load(self, path_to_state_dict : str, device : str):
        '''
        Load previous trained feature fusion model

        Parameters
        ----------
        path_to_state_dict : str
            Path to the save frozen weights in pytorch state dictionary
        device : str
            Device to map the model to. Must be 'cpu' or 'cuda'.
        '''
        self.model.load_state_dict(
                            torch.load(
                                path_to_state_dict, 
                                map_location=torch.device(device),
                                ),
                            )
        self.model.eval()

    
    def process_image(self, img : str | npt.NDArray, img_size : tuple[int, int]) -> torch.Tensor:
        '''
        Prepare image for being fed into the model

        Parameters
        ----------
        img : str | npt.NDArray
            Image to be process as either filepath or array
        image_size : tuple[int, int]
            Size to reshape image to

        Returns
        -------
        img_to_pred : torch.Tensor
            image as tensor ready for model input
        '''
        test_transforms = v2.Compose([
                v2.Resize(img_size),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if isinstance(img, str):
            img_to_pred = Image.open(img).convert('RGB')
        else:
            img_to_pred = Image.fromarray(img)

        img_to_pred = test_transforms(img_to_pred)
        img_to_pred = img_to_pred.unsqueeze(0)

        return img_to_pred
    
    def get_hidden(self, input : str | npt.NDArray, device : str) -> npt.NDArray:
        '''
        Extract the hidden feature representations as a single vector

        Parameters
        ----------
        img : str | npt.NDArray
            Image to be process as either filepath or array
        image_size : tuple[int, int]
            Size to reshape image to

        Returns
        -------
        concat_feat_rep : npt.NDArray
            image as tensor ready for model input
        '''
        self.model.to(device)
        img_to_pred_244 = self.process_image(input, (224, 224))
        img_to_pred_299 = self.process_image(input, (299, 299))
        img_to_pred_244 = img_to_pred_244.to(device)
        img_to_pred_299 = img_to_pred_299.to(device) 

        return self.model.get_hidden(img_to_pred_244, img_to_pred_299).detach().cpu().numpy()[0] 

    def predict(self, img : str | npt.NDArray, device : str) -> tuple:
        '''
        Perform inference using the feature fustion model.

        Parameters
        ----------
        img : str | npt.NDArray
            Image to be process as either filepath or array
        device : str
            Where to map the input image to. Must be 'cpu' or 'cuda'

        Returns
        -------
        values : tuple
            tuple of logit scores and probabilities 
        '''

        def softmax(x): 
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) 
        
        self.model.to(device)
        img_to_pred_244 = self.process_image(img, (224, 224))
        img_to_pred_299 = self.process_image(img, (299, 299))
        img_to_pred_244 = img_to_pred_244.to(device)
        img_to_pred_299 = img_to_pred_299.to(device)
        scores = self.model(img_to_pred_244, img_to_pred_299).detach().cpu().numpy()[0]

        return scores, softmax(scores)
    

class FeatureFusionTrainer():
    '''
    Special Training class to address complications that extend from
    the feature fusion model.

    Attributes
    ----------
    training_csv : str
        path to training csv, must be a 'mixup' style csv. "stardard"
        results can be acheived by make setting lambda to 1 and using
        a place holder image for image_2
    model : FFModel
        instance of the feature fusion model for DNNs
    model_name : str
        template for save trainined models. Must contain special sequence 'XYYX'
    device : str
        Device to use for training. Must be 'cpu' or 'cuda'
    '''
    def __init__(self,
                 training_csv : str,
                 model : FFModel,
                 model_name : str = 'ff_model_XYYX.pkl',
                 device : str = 'cuda',
                 ):
        
        self.train_df = pd.read_csv(training_csv)
        self.model = model
        self.model_name = model_name
        self.device = device
    
    def data_transforms(self, image_size :  tuple[int, int]) -> dict: 
        '''
        Custom image transformers for feature fusion.
        
        Parameters
        ----------
        image_size : tuple[int, int]
            resize image

        Returns
        -------
        transform : dict
            package image transformers to various usecases, by key
        '''

        transform={
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),    
            transforms.RandomApply([
                transforms.RandomRotation(degrees=30)
                ], p=0.5),
            transforms.ColorJitter(
                    brightness=0.2,  
                    contrast=0.1,    
                    saturation=0.5,  
                    hue=0.1          
                ),
            #transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val':transforms.Compose([
            #transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        'post':transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        return transform   

    def train(self,
              num_epochs : int = 20,
              learning_rate : float = 1e-4,
              weight_decay : float = 3e-5,
              batch_size : int = 32) -> None:
        '''
        Training protocal for the feature fusion model.

        Parameters
        ----------
        num_epochs : int
            number of times to run throught training collection
        learning_rate : float
            learning rate to be used in training
        weight_decay : float
            weight decay rate to be used in training
        batch size : float
            number of examples to show the model at each throughput 
        '''
        
        mask = np.random.rand(len(self.train_df)) < 0.8
        train = self.train_df[mask]
        val = self.train_df[~mask]

        transform = self.data_transforms((299,299))
        trainDataset = ISICFeatureFusionMixupDataset(
                            train, 
                            (224,224), 
                            img_size_2=(299,299), 
                            transform=transform['train'],
                            as_tensor=transform['post'])
        valDataset = ISICFeatureFusionMixupDataset(
                            val, 
                            (224,224), 
                            img_size_2=(299,299), 
                            transform=None,
                            as_tensor=transform['post'])

        train_loader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )

        val_loader = DataLoader(
            valDataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=2
        )
    
        optimizer = optim.AdamW(
                            self.model.parameters(), 
                            lr=learning_rate,
                            weight_decay=weight_decay,
                        )
        criterion = nn.CrossEntropyLoss()
        
        iter = 0
        for epoch in range(num_epochs):

            print(f'epoch {epoch+1}/{num_epochs}')

            correct = 0
            total = 0
            for i, (images_244, images_299, labels) in enumerate(tqdm(train_loader)):
                
                if self.device == 'cuda':
                    images_244, images_299, labels = images_244.to('cuda'), images_299.to('cuda'), labels.to('cuda')
                
                optimizer.zero_grad()
                outputs = self.model(images_244, images_299)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()
                iter = iter + 1

            accuracy = 100 * correct / total
            print(f'Train accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}')
            #if epoch % 10 == 0:
                        
            correct = 0
            total = 0
            # Iterate through test dataset
            for images_244, images_299, labels in tqdm(val_loader):

                outputs = self.model(images_244, images_299)
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()

            accuracy = 100 * correct / total
            print(f'Val accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}')
            if self.model_name is None:
                print('model_name is None' )
                torch.save(self.model.state_dict(), f'./saved_models/fusion/feature_fusion_epoch-{epoch}.plk')
            else:
                val_acc = accuracy.detach().numpy()
                print('Saving', self.model_name.replace('XYYX', str(epoch)))
                torch.save(self.model.state_dict(), self.model_name.replace('XYYX', str(epoch)).replace('.plk', f'-val_acc_{val_acc:.4f}.pkl')) 



if __name__ == '__main__':
   
    #Example usage   
    dnn_paths = [
        f'../../saved_models/2025-06-12/dense_net_w_hid_best_mixup_new_negs.pkl',
        f'../../saved_models/2025-06-12/efficient_net_w_hid_best_mixup_new_negs.pkl',
        f'../../saved_models/2025-06-12/inception_w_hid_1_mixup_new_negs.pkl',
        f'../../saved_models/2025-06-12/res_net_w_hid_best_mixup_new_negs.pkl',
    ]
    
    medirai_model = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)
    medirai_model.load(path_to_densenet=dnn_paths[0])
    medirai_model.load(path_to_efficientnet=dnn_paths[1])
    medirai_model.load(path_to_inception=dnn_paths[2])
    medirai_model.load(path_to_resnet=dnn_paths[3])

    medirai_ff_model = MediraiFeatureFusionEnsemble(
        medirai_model.densenet.model,
        medirai_model.efficientnet.model,
        medirai_model.inception.model,
        medirai_model.resnet.model,
        256,
        512,
        device='cpu'
    )

    ff_helper = FFModel(medirai_ff_model)
    ff_helper.load('../../saved_models/ff_large_network/OLD_new_ff_model_19.pkl', device='cpu')
    print(ff_helper.predict('./sample_img.jpg', 'cpu')) 
    #train_csv = './train_data_EQ.csv'
    #train_csv = './mixup_train_data.csv'
    #medirai_ff_trainer = FeatureFusionTrainer(train_csv, medirai_ff_model, device='cpu')
    #medirai_ff_trainer.train()

    #INFERENCE LOOP
    from sklearn.metrics import confusion_matrix
    test_data = pd.read_csv('./test_EQ.csv')
    
    preds = []
    truth = []
    
    for i in range(len(test_data)):

        img_p = test_data.loc[i, 'image_path']
        #img = plt.imread(img_p)

        pred = ff_helper.predict(img_p, device='cpu')#ff_model.predict(img_p)
        #print(pred)

        #print(pred[1][1], '-', test_data.loc[i, 'target'])
        preds.append(pred[1][1])
        truth.append(test_data.loc[i, 'target'])

    preds = (np.array(preds) > 0.5).astype(int)
    truth = np.array(truth)

    print('=========================')
    print(confusion_matrix(truth, preds))
    print('=========================')


