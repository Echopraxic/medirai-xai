
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
from PIL import Image, ImageFilter

from medirai_abstract_model import MediraiAbstractModel as MAM
from resnet_no_reuse import resnet50

class MediraiDensenetHiddenModel(MAM):

    def __init__(
            self,
            config: dict = None
        ):
        
        CONFIG = self._config_precheck(config)
        model = self._fetch_model(CONFIG)
        super().__init__('densenet', model, CONFIG)


    def _config_precheck(self, config):

        if config is not None:
            return config

        CONFIG = {
            'seed': 42,
            'image_size': 224,
            'num_classes': 2,
            'batch_size':32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_fold': 5,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 10,
            'freeze_name_template': './saved_models/dense_net_w_hid_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 3e-6,
            'hidden_size': 256 
        }

        return CONFIG
    
    
    def _fetch_model(self, CONFIG):

        model_ft = models.densenet169(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
                                nn.BatchNorm1d(num_ftrs),
                                nn.Linear(num_ftrs, CONFIG['hidden_size']),
                                nn.ReLU6(),
                                nn.Linear(CONFIG['hidden_size'], CONFIG['num_classes']),
                                ) #Linear(num_ftrs, CONFIG['num_classes'])
        return model_ft
    

    def predict_with_cam(self, src, target_class=None, save_name='./gradcam_explainer.png'):

        target_layer = [self.model.features[-1]]
        grad_img = self.generate_cam(src, target_layer, target_class=target_class, save_name=save_name)
        return grad_img


class MediraiEfficientNetHiddenModel(MAM):

    def __init__(
            self,
            config: dict = None
    ):
        
        CONFIG = self._config_precheck(config)
        model = self._fetch_model(CONFIG)
        super().__init__('efficientnet', model, CONFIG)


    def _config_precheck(self, config):

        if config is not None:
            return config

        CONFIG = {
            'seed': 42,
            'image_size': 224,
            'num_classes': 2,
            'batch_size':32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_fold': 5,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 10,
            'base_outname': 'efficient_net',
            'freeze_name_template': './saved_models/efficient_net_w_hid_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 3e-6,
            'hidden_size':  256,
        }

        return CONFIG
    

    def _fetch_model(self, CONFIG):

        model_ft = models.efficientnet_b6(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        pretrained_layers = [l for l in model_ft.children()]
        num_ftrs = [l for l in pretrained_layers[-1].children()][-1].in_features
        model_ft.classifier[1] = nn.Sequential(
                                    nn.BatchNorm1d(num_ftrs),
                                    nn.Linear(num_ftrs, CONFIG['hidden_size']),
                                    nn.ReLU6(),
                                    nn.Linear(CONFIG['hidden_size'], CONFIG['num_classes']),
                                ) #nn.Linear(num_ftrs, CONFIG['num_classes'])

        return model_ft


    def predict_with_cam(self, src, target_class=None, save_name='./gradcam_explainer.png'):

        target_layer = [self.model.features[-1]]
        grad_img = self.generate_cam(src, target_layer, target_class=target_class, save_name=save_name)
        return grad_img


class MediraiInceptionHiddenModel(MAM):

    def __init__(
            self,
            config: dict = None
    ):

        CONFIG = self._config_precheck(config)
        model = self._fetch_model(CONFIG)
        super().__init__('inception', model, CONFIG)
        

    def _config_precheck(self, config):
       
        if config is not None:
            return config
        
        CONFIG = {
            'seed': 42,
            'image_size': 299,
            'num_classes': 2,
            'batch_size':32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_fold': 5,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 10,
            'base_outname': 'inception',
            'freeze_name_template': './saved_models/inception_w_hid_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 1e-6,
            'hidden_size': 256,

        }

        return CONFIG
    

    def _fetch_model(self, CONFIG):

        model_ft = models.inception_v3(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        
        num_ftrs_aux = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Sequential(
                                    nn.BatchNorm1d(num_ftrs_aux),
                                    nn.Linear(num_ftrs_aux, CONFIG['hidden_size']),
                                    nn.ReLU6(),
                                    nn.Linear(CONFIG['hidden_size'], CONFIG['num_classes']),
                                )#nn.Linear(num_ftrs_aux, CONFIG['num_classes'])

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                                    nn.BatchNorm1d(num_ftrs),
                                    nn.Linear(num_ftrs, CONFIG['hidden_size']),
                                    nn.ReLU6(),
                                    nn.Linear(CONFIG['hidden_size'], CONFIG['num_classes']),
                                )#nn.Linear(num_ftrs, CONFIG['num_classes'])

        return model_ft


    def predict_with_cam(self, src, target_class=None, save_name='./gradcam_explainer.png'):

        for param in self.model.Mixed_7c.parameters():
            param.requires_grad = True

        target_layer = [self.model.Mixed_7c]
        grad_img = self.generate_cam(src, target_layer, target_class=target_class, save_name=save_name)

        for param in self.model.Mixed_7c.parameters():
            param.requires_grad = False

        return grad_img
    

class MediraiResNetHiddenModel(MAM):

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
            'num_classes': 2,
            'batch_size':32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_fold': 5,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 10,
            'base_outname': 'resnet',
            'freeze_name_template': './saved_models/res_net_w_hid_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 3e-6,
            'hidden_size': 256,

        }

        return CONFIG


    def _fetch_model(self, CONFIG):

        model_ft = resnet50(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                                    nn.BatchNorm1d(num_ftrs),
                                    nn.Linear(num_ftrs, CONFIG['hidden_size']),
                                    nn.ReLU6(),
                                    nn.Linear(CONFIG['hidden_size'], CONFIG['num_classes']),
                                ) #nn.Linear(num_ftrs, CONFIG['num_classes'])

        #for param in model_ft.layer4.parameters():
        #    param.requires_grad = True
        
        return model_ft
    

    def predict_with_cam(self, src, target_class=None, save_name='./gradcam_explainer.png'):

        target_layer = [self.model.layer4[-1]]
        grad_img = self.generate_cam(src, target_layer, target_class=target_class, save_name=save_name)
        return grad_img

