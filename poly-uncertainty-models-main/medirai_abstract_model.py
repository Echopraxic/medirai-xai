
from trainer import Trainer
from torchvision.transforms import v2
import torch
from torch import nn, optim
import torch.nn.functional as F

import io
import numpy as np
import numpy.typing as npt
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torchvision
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy

import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

class MediraiAbstractModel:

    '''
    A abstract representation of the DNNs used my MedirAI.

    Attributes
    ----------
    model_name : str
        Name of the model
    model : torchvision.model
            Pretrained model from torchvision
    CONFIG : dict
        Model parameters for training and usage
    mode : str
        Define what mode the model is for eval or training
    feature_extractor : torchvision.feature_extractor
        Place_holder for usecases that require feature extractors downstream.
    '''

    def __init__(self, model_name: str, model, config: dict, mode='train'):
        
        self.model_name: str = model_name
        self.model = model
        self.CONFIG: dict = config
        self.mode: str = mode
        self.feature_extractor:torchvsion.feature_extractor = None


    def load(self, path_to_state_dict:str, device:str) -> None:
        '''
        Load model from frozen state dictionary.

        Parameters
        -----------
        path_to_state_dict: str
            Path as str to frozen weights, must be state dictionry. 
        '''

        self.model.load_state_dict(
                            torch.load(
                                path_to_state_dict, 
                                map_location=torch.device(device),
                                ),
                            )
        self.model.eval()


    def train(self, 
              train_csv: str,
              data_loader: str = 'defualt',
              optimizer='default', 
              criterion='default', 
              scheduler='default',
              balance_classes=True,
              seg_kwargs=None
             ) -> None:
        '''
        Train the DNN.

        Parameters
        ----------
        train_csv: str
            Path to training csv. Training csv must be inline with the
            dataLoader selected.
        
        data_loader : str
            Name of data loader to be used in training

        optimizer:

        '''

        assert self.mode == 'train'

        def balance_train_df(train_df: pd.DataFrame) -> pd.DataFrame:
            
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
            data_loader=data_loader,
            seg_kwargs=seg_kwargs
        )

        trainer.fit()

    def read_image(self, img: str | npt.NDArray) -> Image.Image:
        '''
        Read image from arry or path.

        Parameters
        ----------
        img: str | np.array
            Path to image or image as array.

        Returns
        -------
        img: PIL Image
            Loaded image as a PIL image.
        '''

        if isinstance(img, str):
            img_to_pred = Image.open(img).convert('RGB')
        else:
            img_to_pred = Image.fromarray(img)

        return img_to_pred

    def process_image(self, img: Image.Image) -> Image.Image:
        '''
        Process image for model inference.

        Parameters
        ----------
        img: PIL Image
            Image to make ready for inference

        Returns
        -------
        img: PIL Image
            Image ready for inference
        '''
        test_transforms = v2.Compose([
                v2.Resize((self.CONFIG['image_size'], self.CONFIG['image_size'])),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        img_to_pred = self.read_image(img)
        img_to_pred = test_transforms(img_to_pred)
        img_to_pred = img_to_pred.unsqueeze(0)

        return img_to_pred

    
    def predict(self, input: npt.NDArray | str, device: str) -> tuple:
        '''
        Do inference call.

        Parameters
        ----------
        input: np.array | str
            Either path to image or image as array.
        
        Returns
        -------
        tuple: (logits, probabilities)
            Binary classification scores as logits and probabilites.
       
        '''
        def softmax(x): 
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) 
        
        self.model.to(device)
        img_to_pred = self.process_image(input)
        img_to_pred = img_to_pred.to(self.CONFIG['device'])
        #scores = self.model(img_to_pred).detach().numpy()[0]
        scores = self.model(img_to_pred).detach().cpu().numpy()[0]

        return scores, softmax(scores)
    
    def set_parameter_requires_grad(self, model, feature_extracting):

        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    
    def generate_cam(self, img: str | npt.NDArray, 
                     target_class: int, 
                     target_layer: list[str], 
                     save_name: str = './gradcam_explainer.png'):
        '''
        Produce the images for GradCAM explainability.

        Pamaters
        --------
        img : str | npt.NDArray
            Image to apply GradCAM explainability on.
        target_class : int
            GradCAM required a priori knowledge of the class of the image in question.
        target_layer : int
            Layer of the network we want to extract GradCAM from.
        '''

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
        #grad_images = copy.deepcopy(Image.fromarray(images.astype(np.uint8)))
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
        fig.suptitle('GradCAM Explainability Results')
        axs[0].imshow(raw_img)
        axs[0].set_title('Image')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(cam)
        axs[1].set_title('Heat Map')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        axs[2].imshow(cam_image)
        axs[2].set_title('Overlay')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        
        if save_name is not None:
            plt.savefig(save_name)
        
    def predict_with_shap(self, src : str, save_name : str) -> None:
        '''
        Helper function for SHAP explainability to keep API consistent

        Parameters
        ----------
        src : str
            path to image to perform SHAP on
        save_name : str
            path where explainability figure will be saved
        '''
        self.generate_shap(src, save_name)

    def predict_with_lime(self, src : str, save_name : str) -> None:
        '''
        Helper function for LIME explainability to keep API consitent

        Parameters
        ----------
        src : str
            path to image to perform LIME on
        save_name : str
            path where explainability figure will be saved
        '''
        self.generate_lime(src, save_name)
   
    def generate_shap(self, img : str, save_name : str = './shap_explainer.png') -> None:
        '''
        Generate the SHAP explainability figure. Original publication:
        https://arxiv.org/abs/1705.07874

        Parameters
        ----------
        src : str
            path to image to perform SHAP on
        save_name
            path where explainability figure will be saved
        '''
       
        def predict(img: np.ndarray) -> torch.Tensor:
            imgs = [self.process_image(i) for i in img]
            img = torch.cat(imgs)
            output = self.model(img)
            return output
        
        X = self.read_image(img)
        X = X.resize( (self.CONFIG['image_size'], self.CONFIG['image_size']) )
        X = np.array([X]).astype(np.uint8)

        masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)
        class_names = ['benign','malignant']
        explainer = shap.Explainer(predict, masker_blur, output_names=class_names)
        shap_values = explainer(
            X,
            max_evals=500,
            batch_size=50,
            outputs=shap.Explanation.argsort.flip,
        )
        
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

        shap.image_plot(    
            shap_values=shap_values.values,
            pixel_values=shap_values.data[0],
            labels=shap_values.output_names,
            true_labels='',
            show=False
        )
        
        plt.suptitle('SHAP Explainability Results')
        
        if save_name is not None:
            plt.savefig(save_name)

    def generate_lime(self, img, save_name='lime_explainer.png') -> None:
        '''
        Generate the LIME explainability figure. Original publication:
        https://arxiv.org/abs/1602.04938

        Parameters
        ----------
        src : str
            path to image to perform LIME on
        save_name
            path where explainability figure will be saved
        '''
        
        def predict(img: np.ndarray) -> torch.Tensor:
            imgs = [self.process_image(i) for i in img]
            img = torch.cat(imgs)
            output = self.model(img)
            output = F.softmax(output, dim=1) 
            return output.detach().cpu().numpy()
        
        X = self.read_image(img)
        X = X.resize( (self.CONFIG['image_size'], self.CONFIG['image_size']) )
        X = np.array(X)
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
                X,
                predict,
                hide_color=0,
                num_samples=600
                )

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=False,
                                                    num_features=5,
                                                    hide_rest=False)
        img_boundry = mark_boundaries(temp/255.0, mask)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        fig.suptitle('LIME Explainability Results')
        axs[0].imshow(X)
        axs[0].set_title('Image')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(img_boundry)
        axs[1].set_title('Contributing Features')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        if save_name is not None:
            plt.savefig(save_name)

    def lock_grads(self) -> None:
        '''
        Helper function to lock gradients and prevent training
        '''

        for _, param in self.model.named_parameters():
            param.requires_grad = False
        
        self.mode = 'inference'
    

    def unlock_grads(self) -> None:
        '''
        Helper function to unlock gradients and allow training
        '''
        for _, param in self.model.named_parameters():
            param.requires_grad = True
        
        self.mode = 'train'


    def unlock_n_leading_grads(self, n : int = 3) -> None:
        '''
        Unlock the leading n layers of a model.

        Paramters:
        ----------
        n : int
            number of leading n layers of the model.
        '''
        params = [param for param in self.model.parameters()]
        for param in params[:n]:
            param.requires_grad = True

        self.mode = 'train'


    def unlock_n_trailing_grads(self, n=3):
        
        params = [param for param in self.model.parameters()]
        for param in params[:-n]:
            param.requires_grad = True

        self.mode = 'train'


    def set_feature_extractor(self, layer: str | None) -> None:
        '''
        Helper function to create a pytorch feature extractor.

        Parameters
        ----------
        layer: str
            layer of model at which to create the feature extractor
        '''

        if layer is None:

            self.feature_extractor = None
        
        else:
            self.feature_extractor = create_feature_extractor(
                self.model,
                return_nodes=[layer]
            )

    def unlock_named_layers(self, names : list[str]) -> None:
        '''
        Unlock layers in torchvision models by name. Modifies
        self.model in place

        Paramters
        ---------
        names : list[str]
            list of layer names to unlock
        '''

        self.lock_grads()
        
        for layer_name in names:
            for n, p in self.model.named_parameters():
                if layer_name in n:
                    p.requires_grad = True

        self.mode = 'train'

    def get_feature_representation(self, img: torch.Tensor, layer: str) -> torch.Tensor:
        '''
        Using a previously create feature extractor, get the features.

        Parameters
        ----------
        img: torch.Tensor
            Single image to get feature representations of downstream in the model

        Returns
        -------
        feat_rep: torch.Tensor
            Feature representation at provided layer of provided image.
        '''
        #img = self.process_image(img) 
        feat_rep = self.feature_extractor(img.clone())[layer]
        return feat_rep

    
    

