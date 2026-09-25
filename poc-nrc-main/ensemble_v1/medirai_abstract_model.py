
from trainer import Trainer
from torchvision.transforms import v2
import torch
from torch import nn, optim

import numpy as np
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy

class MediraiAbstractModel:

    def __init__(self, model_name, model, config, mode='train'):
        
        self.model_name = model_name
        self.model = model
        self.CONFIG = self._validate_config(config)
        self.mode = mode
        self.feature_extractor = None


    def load(self, path_to_state_dict, device):

        self.model.load_state_dict(
                            torch.load(
                                path_to_state_dict, 
                                map_location=torch.device(self.CONFIG['device']),
                                ),
                            )
        self.model.eval()


    def train(self, 
              train_csv, 
              optimizer='default', 
              criterion='default', 
              scheduler='default',
              balance_classes=True, 
             ):

        assert self.mode == 'train'

        def balance_train_df(train_df):
            
            class_counts = len(train_df.target.unique())
            class_counts = train_df['target'].value_counts()
            min_class_count = min(class_counts)

            balanced_dfs = []
            for label in class_counts.index:

                if class_counts[label] > min_class_count:
                    df_class = train_df[train_df['target'] == label].sample(min_class_count, random_state=self.CONFIG['seed'])
                else:
                    df_class = train_df[train_df['target'] == label]
        
                balanced_dfs.append(df_class)

            train_dataset = pd.concat(balanced_dfs, axis=0)
            train_dataset = train_dataset.sample(frac=1, random_state=self.CONFIG['seed']).reset_index(drop=True)

            return train_dataset


        train_df = pd.read_csv(train_csv)
        print(train_df.columns)
        #assert set(train_df.columns) == set(['image_id', 'target', 'image_path'])
        
        if balance_classes:
            train_df = balance_train_df(train_df)

        if criterion == 'default':
            criterion = nn.CrossEntropyLoss()

        if optimizer == 'default':
            optimizer = optim.AdamW(
                            self.model.parameters(), 
                            lr=self.CONFIG['learning_rate'], 
                            weight_decay=self.CONFIG['weight_decay'],
                        )
        
        if scheduler == 'default':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode='min',
                            factor=0.1,
                            patience=3,
                            verbose=True,
                        )
        
        assert criterion is not None
        assert optimizer is not None

        trainer = Trainer(
            self.model_name,
            self.model,
            self.CONFIG,
            criterion,
            optimizer,
            train_df,
            scheduler=scheduler,
            device=self.CONFIG['device'],
        )

        trainer.fit()


    def process_image(self, img):

        test_transforms = v2.Compose([
                v2.Resize((self.CONFIG['image_size'], self.CONFIG['image_size'])),
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

    
    def predict(self, input, device):

        def softmax(x): 
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) 
        
        self.model.to(device)
        img_to_pred = self.process_image(input)
        img_to_pred = img_to_pred.to(device)
        #scores = self.model(img_to_pred).detach().numpy()[0]
        scores = self.model(img_to_pred).detach().cpu().numpy()[0]

        return scores, softmax(scores)
    

    def _validate_config(self, config):

        return config
    
    
    def set_parameter_requires_grad(self, model, feature_extracting):

        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    
    def generate_cam(self, img, target_class, target_layer) -> Image:

        if self.mode != 'train': #gradcam requires unlocked model gradients
            self.unlock_grads() 

        input_tensor = self.process_image(img)#preprocess_image(img)
        raw_img = Image.open(img).convert('RGB')
        raw_img = raw_img.resize((self.CONFIG['image_size'], self.CONFIG['image_size']))
        raw_img = np.array(raw_img)

        targets = [ClassifierOutputTarget(target_class)]
        with GradCAM(model=self.model, target_layers=target_layer) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            cam_image = show_cam_on_image(raw_img/255, grayscale_cams[0, :], use_rgb=True)
        
        cam = np.uint8(255 * grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((raw_img, cam, cam_image)).astype(np.float32)
        grad_images = copy.deepcopy(Image.fromarray(images.astype(np.uint8)))
        
        return np.array(grad_images)


    def lock_grads(self):

        for _, param in self.model.named_parameters():
            param.requires_grad = False
        
        self.mode = 'inference'
    

    def unlock_grads(self):

        for _, param in self.model.named_parameters():
            param.requires_grad = True
        
        self.mode = 'train'


    def unlock_n_leading_grads(self, n=3):
        
        params = [param for param in self.model.parameters()]
        for param in params[:n]:
            param.requires_grad = True

        self.mode = 'train'


    def unlock_n_trailing_grads(self, n=3):
        
        params = [param for param in self.model.parameters()]
        for param in params[:-n]:
            param.requires_grad = True

        self.mode = 'train'


    def set_feature_extractor(self, layer):

        if layer is None:

            self.feature_extractor = None
        
        else:
            self.feature_extractor = create_feature_extractor(
                self.model,
                return_nodes=[layer]
            )

    def unlock_named_layers(self, names):

        self.lock_grads()
        
        for layer_name in names:
            for n, p in self.model.named_parameters():
                if layer_name in n:
                    p.requires_grad = True

        self.mode = 'train'

    def get_feature_representation(self, img, layer):
        
        #img = self.process_image(img) 
        feat_rep = self.feature_extractor(img.clone())[layer]
        return feat_rep

    
    

