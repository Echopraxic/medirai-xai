
import numpy as np
import numpy.typing as npt
import torch
import os
import matplotlib.pyplot as plt
import warnings
import time
import typing

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from medirai_models_w_hidden import MediraiDensenetHiddenModel, MediraiEfficientNetHiddenModel, MediraiInceptionHiddenModel, MediraiResNetHiddenModel

class MediraiEnsembleModelV1:
    '''
    Class to wrap all DNN use cases.

    Attributes
    ----------
    device : str
        Device where pytorch models should reside. Must be 'cuda' or 'cpu'.
    densenet : torchvision.model.densenet169
        Instance of densenet model
    efficientnet : torchvision.model.efficientnet_b6
        Instance of efficientnet model
    inception : torchvision.model.inception_v3
        Instance of inception model
    resnet : resnet_no_reuse.resnet50
        Instance of resnet model
    '''
    def __init__(self, 
                 device : str ='cuda', 
                 use_hidden_layers=True,
                 configs=[None, None, None, None]):
                 

        #self._load_medirai_models(use_hidden_layers)
        self.device = device
        self._set_seed()

        if use_hidden_layers:
            self.densenet = MediraiDensenetHiddenModel(config=configs[0]) 
            self.efficientnet = MediraiEfficientNetHiddenModel(config=configs[1]) 
            self.inception = MediraiInceptionHiddenModel(config=configs[2])
            self.resnet = MediraiResNetHiddenModel(config=configs[3])
        else:
            raise (f'Use of models without hidden MLP layers is depricated.')

        self.models_loaded = [False, False, False, False]
        

    def __repr__(self):
        return f'DenseNet (loaded: {self.models_loaded[0]})\nEfficientNet (loaded: {self.models_loaded[1]})\nInception (loaded: {self.models_loaded[2]}) \nResNet (loaded: {self.models_loaded[3]})'


    def _set_seed(self, seed: int = 42) -> None:
        '''
        Set all seeds for reproducibility.

        Parameters
        Seed : int
            Random seed value
        '''
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def train(self, 
              training_csv : str,
              to_train: str ='all',
              data_loader='default',
              **kwargs) -> None:
        '''
        Training protocol to train all DNNs.

        Parameters
        ----------
        train_csv : str
            path to training csv. Columns of training csv must match data loader.
        to_train : str
            specify which models should be trained, 'all' will train all 4 models in series.
        data_loader : str
            specify which data data loader to use. Data loader must match training csv.
        **kwargs
            additional arguments are held here. These are only used when the segmentation 
            data loader is in use 
        '''       

        if len(kwargs) == 0:
            seg_kwargs = None
        else:
            seg_kwargs = dict(kwargs)

        if self.device == 'cpu':
            warnings.warn('Training on CPU with take a significant amount of time. CUDA is recommended.')
            time.sleep(3)

        if to_train == 'all':
            warnings.warn('All models will be trained sequentially. This will take a significant amount of time.')
            time.sleep(3)
        
        if to_train == 'all' or 'densenet' in to_train:
            self.densenet.train(training_csv, data_loader=data_loader, seg_kwargs=seg_kwargs)
        
        if to_train == 'all' or 'efficientnet' in to_train:
            self.efficientnet.train(training_csv, data_loader=data_loader, seg_kwargs=seg_kwargs)
        
        if to_train == 'all' or 'inception' in to_train:
            self.inception.train(training_csv, data_loader=data_loader, seg_kwargs=seg_kwargs)
        
        if to_train == 'all' or 'resnet' in to_train:
            self.resnet.train(training_csv, data_loader=data_loader, seg_kwargs=seg_kwargs)
        

    def load(self, 
             path_to_densenet: typing.Optional[str]=None,
             path_to_efficientnet: typing.Optional[str]=None,
             path_to_inception: typing.Optional[str]=None,
             path_to_resnet: typing.Optional[str]=None) -> None:
        '''
        Load individual frozen weights for the four networks. Must be 
        provided as a pytorch state dictionary. 

        Parameters
        ----------
        path_to_densenet : str
            Path to frozen densenet weights
        path_to_efficientnet : str
            Path to frozen efficition net weights
        path_to_inception : str
            path to frozen inception weights
        path_to_resnet : str
            path to frozen resnet weights
        ''' 
        #load_models
        if path_to_densenet is not None:
            self.densenet.load(path_to_densenet, self.device)
            self.models_loaded[0] = True

        if path_to_efficientnet is not None:
            self.efficientnet.load(path_to_efficientnet, self.device)
            self.models_loaded[1] = True

        if path_to_inception is not None:
            self.inception.load(path_to_inception, self.device)
            self.models_loaded[2] = True

        if path_to_resnet is not None:
            self.resnet.load(path_to_resnet, self.device)
            self.models_loaded[3] = True
        
    def predict(self, img : str | npt.NDArray) -> npt.NDArray:
        
        '''
        Perform inference call using all four networks.

        Parameters
        ----------
        img : str | npt.NDArray
            source image to perform inference call on, can be array or path to image.

        Returns
        -------
        (probabilities, predictions) : tuple[tuple[float]]
            tuple containing probabilities of benign and malignant, and binary predictions
            as 0, 1. Will return in format of [[prob_1, ..., prob_4], [pred_1, ..., pred_4]]
        '''
        #img = self.val_img_and_load(src)

        #h, w, c = img.shape
        #img_with_batch = (1, c, h, w)
        #img = np.reshape(img, img_with_batch)
        
        densenet_scores, densenet_pred = self.densenet.predict(img, self.device)
        efficientnet_scores, efficientnet_pred = self.efficientnet.predict(img, self.device) 
        inception_scores, inception_pred = self.inception.predict(img, self.device) 
        resnet_scores, resnet_pred = self.resnet.predict(img, self.device)

        malig_probs = [densenet_pred[1],
                       efficientnet_pred[1],
                       inception_pred[1],
                       resnet_pred[1]]
        malig_probs = np.array(malig_probs)

        malig_preds = [np.argmax(densenet_pred),
                       np.argmax(efficientnet_pred),
                       np.argmax(inception_pred),
                       np.argmax(resnet_pred)]
        malig_preds = np.array(malig_preds)
        
        return np.array([malig_probs, malig_preds])

if __name__ == '__main__':
    
    # Example usage:
    print('Model with Hidden Layers:')
    model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)

    print('Loading Pretained Models...')    
    model_with_hidden.load(path_to_densenet='../saved_models/dnns_mixup_3M/dense_net_w_hid_best_mixup_3M.pkl')
    model_with_hidden.load(path_to_efficientnet='../saved_models/dnns_mixup_3M/efficient_net_w_hid_best_mixup_3M.pkl')
    model_with_hidden.load(path_to_inception='../saved_models/dnns_mixup_3M/inception_w_hid_best_mixup_3M.pkl')
    model_with_hidden.load(path_to_resnet='../saved_models/dnns_mixup_3M/res_net_w_hid_best_mixup_3M.pkl')
    print(model_with_hidden)

    sample_img, sample_is_malig = './sample_data/sample_img.jpg', 0

    probs, preds = model_with_hidden.predict(sample_img)
    print('Model Predictions:')
    print(preds)
    print('Model Probs: ')
    print(probs)
    
    #Create data mixup train csv:
    from make_mixup_data import make_mixup_data_train_csv
    train_csv_no_mixup = './train_data_EQ.csv' #needs equal number of class observations

    #Needs be ran once to create the mixup csv, after can use filepath
    train_csv_w_mixup = make_mixup_data_train_csv(train_csv_no_mixup)
    #train_csv_w_mixup = './mixup_train_data.csv'
    
    model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)
    model_with_hidden.train(train_csv_w_mixup, data_loader='mixup')

