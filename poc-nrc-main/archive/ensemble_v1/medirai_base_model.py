
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import warnings
import time

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from medirai_models_w_hidden import MediraiDensenetHiddenModel, MediraiEfficientNetHiddenModel, MediraiInceptionHiddenModel, MediraiResNetHiddenModel
from medirai_models import MediraiDensenetModel, MediraiEfficientNetModel, MediraiInceptionModel, MediraiResNetModel


class MediraiEnsembleModelV1:

    def __init__(self, 
                 device='cuda', 
                 use_hidden_layers=True,
                 configs=[None, None, None, None]):

        self._load_medirai_models(use_hidden_layers)
        self.device = device
        self._set_seed()

        if use_hidden_layers:
            self.densenet = MediraiDensenetHiddenModel(config=configs[0]) 
            self.efficientnet = MediraiEfficientNetHiddenModel(config=configs[1]) 
            self.inception = MediraiInceptionHiddenModel(config=configs[2])
            self.resnet = MediraiResNetHiddenModel(config=configs[3])
        else:
            self.densenet = MediraiDensenetModel(config=configs[0]) 
            self.efficientnet = MediraiEfficientNetModel(config=configs[1]) 
            self.inception = MediraiInceptionModel(config=configs[2])
            self.resnet = MediraiResNetModel(config=configs[3]) 

        self.models_loaded = [False, False, False, False]

    def __repr__(self):
        return f'DenseNet (loaded: {self.models_loaded[0]})\nEfficientNet (loaded: {self.models_loaded[1]})\nInception (loaded: {self.models_loaded[2]}) \nResNet (loaded: {self.models_loaded[3]})'


    def _load_medirai_models(self, use_hidden_layers) -> None:

        print('Doing model imports...')
        

    def _set_seed(self, seed=42):
        '''
        Set all seeds for reproducibility.
        '''
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


    def train(self, 
              training_csv,
              to_train: str ='all',):
        
        if self.device == 'cpu':
            warnings.warn('Training on CPU with take a significant amount of time. CUDA is recommended.')
            time.sleep(3)

        if to_train == 'all':
            warnings.warn('All models will be trained sequentially. This will take a significant amount of time.')
            time.sleep(3)

        if to_train == 'all' or 'densenet' in to_train:
            self.densenet.train(training_csv)
        
        if to_train == 'all' or 'efficientnet' in to_train:
            self.efficientnet.train(training_csv)
        
        if to_train == 'all' or 'inception' in to_train:
            self.inception.train(training_csv)
        
        if to_train == 'all' or 'resnet' in to_train:
            self.resnet.train(training_csv)
        

    def load(self, 
             path_to_densenet=None, 
             path_to_efficientnet=None,
             path_to_inception=None,
             path_to_resnet=None):
        
        #load_models
        if path_to_densenet is not None:
            self.densenet.load(path_to_densenet, self.device)
            self.models_loaded[0] = True

        if path_to_efficientnet is not None:
            self.efficientnet.load(path_to_efficientnet, self.device)
            #self.efficientnet.model = self.efficientnet.model.module
            self.models_loaded[1] = True

        if path_to_inception is not None:
            self.inception.load(path_to_inception, self.device)
            self.models_loaded[2] = True

        if path_to_resnet is not None:
            self.resnet.load(path_to_resnet, self.device)
            self.models_loaded[3] = True
        

    def val_img_and_load(self, src):
        
        if not all(self.models_loaded):
            raise RuntimeError("Not all models are loaded, 'predict' functionality unavailable.")
        
        img = None
        if isinstance(src, str):
            os.path.exists(src)
            try:
                img = plt.imread(src)
            except OSError:
                print(f'Image located at {src} could not be read!')
        
        else:
            try:
                img = np.array(src)
            except TypeError:
                print(f'Img received as type {type(src)} could not be cast to np.array!')

        #print(img)
        return img.astype(np.float32)

    def predict(self, img: nd.array):

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

        return malig_probs, malig_preds

    
    def predict_with_cam(self, img, label, save_path=None):

        #img = self.val_img_and_load(src)/255.0

        densenet_cam_pred = self.densenet.predict_with_cam(img, label)
        efficient_net_cam = self.efficientnet.predict_with_cam(img, label)
        inception_cam = self.inception.predict_with_cam(img, label)
        resnet_cam = self.resnet.predict_with_cam(img, label)

        malig_probs, malig_preds = self.predict(img)

        fig, axs = plt.subplots(4, figsize=(5,8))
        fig.suptitle('GradCAM Results')
        axs[0].imshow(densenet_cam_pred)
        axs[1].imshow(efficient_net_cam)
        axs[2].imshow(inception_cam)
        axs[3].imshow(resnet_cam)

        model_names = ['DenseNet', 'EfficientNet','Inception','ResNet']
        for i in range(4):
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(f'{model_names[i]} is malignant: {malig_preds[i]} ({malig_probs[i]:.5f})')

        if save_path is None:
            plt.show()
        else: 
            plt.savefig(save_path)


if __name__ == '__main__':
    
    '''
    #==================================================
    This code requires changes to trainer.py to run.

    Will be resolved in upcoming API upgrades.
    #==================================================
    sample_img, sample_is_malig = './sample_img.jpg', 0

    print('Model without Hidden Layers:')
    model_no_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=False)
    #print('Performing Training Tests')
    #training_csv = './train_data_TESTING.csv'
    #model_no_hidde.train(training_csv, to_train='all')
    
    print('Loading Pretained Models...')    
    model_no_hidden.load(path_to_densenet='../../saved_models/medirai_models_without_hidden/dense_net_best.pkl')
    model_no_hidden.load(path_to_efficientnet='../../saved_models/medirai_models_without_hidden/efficient_net_best.pkl')
    model_no_hidden.load(path_to_inception='../../saved_models/medirai_models_without_hidden/inception_best.pkl')
    model_no_hidden.load(path_to_resnet='../../saved_models/medirai_models_without_hidden/res_net_best.pkl')
    print(model_no_hidden)

    
    probs, preds = model_no_hidden.predict(sample_img)
    print('Model Predictions:')
    print(preds)
    print('Model Probs: ')
    print(probs)

    pred_cam_img = model_no_hidden.predict_with_cam(sample_img, sample_is_malig, save_path='./grad_cam_results/test_image_from_model_without_hidden.png')
    
    print('Model with Hidden Layers:')
    model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)

    print('Loading Pretained Models...')    
    model_with_hidden.load(path_to_densenet='../../saved_models/medirai_models_with_hidden/dense_net_w_hid_256_best.pkl')
    model_with_hidden.load(path_to_efficientnet='../../saved_models/medirai_models_with_hidden/efficient_net_w_hid_256_best.pkl')
    model_with_hidden.load(path_to_inception='../../saved_models/medirai_models_with_hidden/inception_w_hid_256_best.pkl')
    model_with_hidden.load(path_to_resnet='../../saved_models/medirai_models_with_hidden/res_net_w_hid_256_best.pkl')
    print(model_with_hidden)

    sample_img, sample_is_malig = './sample_img.jpg', 0

    probs, preds = model_with_hidden.predict(sample_img)
    print('Model Predictions:')
    print(preds)
    print('Model Probs: ')
    print(probs)

    pred_cam_img = model_with_hidden.predict_with_cam(sample_img, sample_is_malig, save_path='./grad_cam_results/test_image_from_model_with_hidden.png')
    '''
    
    from make_mixup_data import make_mixup_data_train_csv
    #Create data mixup train csv:
    train_csv_no_mixup = './train_data_EQ.csv' #needs equal number of class observations

    #Needs be ran once to create the mixup csv, after can use filepath
    train_csv_w_mixup = make_mixup_data_train_csv(train_csv_no_mixup)
    #train_csv_w_mixup = './mixup_train_data.csv'
    
    model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)
    model_with_hidden.train(train_csv_w_mixup)  


