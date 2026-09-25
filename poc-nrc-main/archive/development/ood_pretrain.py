
import pandas as pd

from medirai_abstract_model import MediraiAbstractModel as MAM

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
from PIL import Image, ImageFilter

class MediraiWideResNetHiddenModel(MAM):

    def __init__(
            self,
            config: dict = None
    ):
        
        CONFIG = self._config_precheck(config)
        model = self._fetch_model(CONFIG)
        super().__init__('resnet', model, CONFIG)


    def _config_precheck(self, config):

        if config is not None:
            return config

        CONFIG = {
            'seed': 42,
            'image_size': 224,
            'num_classes': 10,
            'batch_size':32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_fold': 5,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 10,
            'base_outname': 'resnet',
            'freeze_name_template': './saved_models/wide_res_net_w_hid_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 0.01,
            'hidden_size': 256,

        }

        return CONFIG


    def _fetch_model(self, CONFIG):

        model_ft = models.wide_resnet50_2(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, CONFIG['hidden_size']),
                                    nn.ReLU6(),
                                    nn.Linear(CONFIG['hidden_size'], CONFIG['num_classes']),
                                ) #nn.Linear(num_ftrs, CONFIG['num_classes'])

        #for param in model_ft.layer4.parameters():
        #    param.requires_grad = True
        
        return model_ft
    

    def predict_with_cam(self, src, label):
 
        target_layer = [self.model.layer4[-1]]
        grad_img = self.generate_cam(src, label, target_layer)
        return grad_img
    
if __name__ == '__main__':

    model =  MediraiWideResNetHiddenModel()
    train_df = pd.read_csv('./biomed_50k_percentile_bin.csv')
    train_df = train_df[['image_id','image_path','percentile_bin']]
    train_df.columns =['image_id','image_path','target'] 
    train_df.to_csv('./ood_training.csv', index=False)
    print(train_df.head())

    model.train('./ood_training.csv', balance_classes=False)
