
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models

import resnet_no_reuse
from medirai_abstract_model import MediraiAbstractModel as MAM

class MediraiDensenetModel(MAM):

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
            'n_fold': 10,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 100,
            'freeze_name_template': './saved_models_TESTS/dense_net_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 0.01
        }

        return CONFIG
    
    def _fetch_model(self, CONFIG):

        model_ft = models.densenet169(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, CONFIG['num_classes'])
        return model_ft
    

    def predict_with_cam(self, src, label):

        target_layer = [self.model.features[-1]]
        grad_img = self.generate_cam(src, label, target_layer)
        return grad_img


class MediraiEfficientNetModel(MAM):

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
            'n_fold': 10,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 100,
            'base_outname': 'efficient_net',
            'freeze_name_template': './saved_models/efficient_net_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 0.01
        }

        return CONFIG
    
    def _fetch_model(self, CONFIG):

        model_ft = models.efficientnet_b6(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        pretrained_layers = [l for l in model_ft.children()]
        num_ftrs = [l for l in pretrained_layers[-1].children()][-1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, CONFIG['num_classes'])

        return model_ft


    def predict_with_cam(self, src, label):
        
        target_layer = [self.model.features[-1]]
        grad_img = self.generate_cam(src, label, target_layer)
        return grad_img

    
class MediraiInceptionModel(MAM):

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
            'n_fold': 10,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 100,
            'base_outname': 'inception',
            'freeze_name_template': './saved_models/inception_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 0.01
        }

        return CONFIG
    

    def _fetch_model(self, CONFIG):

        model_ft = models.inception_v3(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        
        num_ftrs_aux = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs_aux, CONFIG['num_classes'])

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, CONFIG['num_classes'])

        return model_ft


    def predict_with_cam(self, src, label):
        
        for param in self.model.Mixed_7c.parameters():
            param.requires_grad = True
            
        target_layer = [self.model.Mixed_7c]
        grad_img = self.generate_cam(src, label, target_layer)

        for param in self.model.Mixed_7c.parameters():
            param.requires_grad = False

        return grad_img
    

class MediraiResNetModel(MAM):

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
            'n_fold': 10,
            'learning_rate': 1e-5,
            'gamma': 0.95,
            'momentum': 0.9,
            'num_epochs': 100,
            'base_outname': 'resnet',
            'freeze_name_template': './saved_models/res_net_XYYX.pkl',
            'grad_acc': 8,
            'weight_decay': 0.01
        }

        return CONFIG


    def _fetch_model(self, CONFIG):

        #model_ft = models.resnet50(pretrained=True)
        model_ft = resnet_no_reuse.resnet50(pretrained=True)
        #self.set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, CONFIG['num_classes'])

        for param in model_ft.layer4.parameters():
            param.requires_grad = True
        
        return model_ft
    

    def predict_with_cam(self, src, label):
        
        target_layer = [self.model.layer4[-1]]
        grad_img = self.generate_cam(src, label, target_layer)
        return grad_img