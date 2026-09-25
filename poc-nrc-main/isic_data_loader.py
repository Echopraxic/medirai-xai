import cv2
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision 
from albumentations.core.transforms_interface import ImageOnlyTransform

from typing import Optional

class ISICDataset(Dataset):
    '''
    Vanilla Data loader. Data must be passed as cvs with columns
    'image_path', 'target', and 'image_id'.

    Attributes
    ----------
    df : pd.DataFrame
        training data layed out in a csv. Must have columns
        'image_path', 'target', 'image_id'
    size : tuple[int, int]
        size to reshape images to as required by the mdoel
    transform : torchvision.transforms.Compose
        traning augmentations to aid in training
    '''

    def __init__(self, df: pd.DataFrame, size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None, blur_radius : float = 0):

        super().__init__()
        self.data = df.reset_index(drop=True)
        self.size = size
        self.transform=transform
        self.blur_radius = blur_radius
        self.required_cols = set(['image_path', 'target', 'image_id'])

       
    def _validate_cols(self) -> None:
        '''
        Validate the manditory columns are present
        ''' 
        
        df_cols = set(self.data.columns)
        col_diff = list(self.required_cols.difference(df_cols))

        if len(col_diff) > 0:
            col_diff = ['"'+str(c)+'"' for c in col_diff]
            raise AttributeError(f'Reference DataFrame has incorrect columns. Columns: {','.join([c for c in col_diff])} are Missing.')
     
        
    def __len__(self) -> int:

        return len(self.data)


    def __getitem__(self, idx):

        file_path = self.data.loc[idx, 'image_path']
        label = self.data.loc[idx, 'target']
        image_id = self.data.loc[idx, 'image_id']

        img = Image.open(file_path).convert('RGB')

        # Blur used to be applied only to 'melanoma' (Kaggle, ~all malignant) images, handing the model a
        # "blurry => malignant" shortcut that inference never sees (CODEBASE_TODO P0-7). If blur is wanted
        # for domain matching it must be applied to every image, and identically at inference.
        if self.blur_radius > 0:
            img = self.gauss_blur(img)

        if self.transform:
            img = self.transform(img)

        return img, label
    

    def gauss_blur(self, img : Image.Image) -> Image.Image:
        '''
        Apply Gaussian blur to an image

        Parameters
        ----------
        img : Image.Image
            image to apply the blur to

        Returns
        -------
        img : Image.Image
            image with blur
        '''

        return img.filter(ImageFilter.GaussianBlur(radius = self.blur_radius))


class ISICDatasetWMod(Dataset):
    '''
    Data Loader to artifically increase the size of the training
    set using deterministic modifications. Modificatio must be listed
    in training csv passed. Modification must be one of 'rot1', 'rot2'
    'rot3', 'mir_h', 'mir_v', 'transpose','None'.

    Attributes
    ----------
    df : pd.DataFrame
        training data layed out in a csv. Must have columns
        'image_path', 'target', 'image_id', 'mod'
    size : tuple[int, int]
        size to reshape images to as required by the mdoel
    transform : torchvision.transforms.Compose
        traning augmentations to aid in training
    '''

    def __init__(self, df: pd.DataFrame, size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None):

        super().__init__()
        self.data = df
        self.size = size
        self.transform=transform
        self.required_cols = set(['image_path', 'target', 'image_id','mod'])
        self._validate_cols()

    def _validate_cols(self) -> None:
        '''
        Validate the manditory columns are present
        '''
        df_cols = set(self.data.columns)
        col_diff = list(self.required_cols.difference(df_cols))

        if len(col_diff) > 0:
            col_diff = ['"'+str(c)+'"' for c in col_diff]
            raise AttributeError(f'Reference DataFrame has incorrect columns. Columns: {','.join([c for c in col_diff])} are Missing.')
     
    def __len__(self):

        return len(self.data)
    
    
    def __getitem__(self, idx):

        file_path = self.data.loc[idx, 'image_path']
        label = self.data.loc[idx, 'target']
        image_id = self.data.loc[idx, 'image_id']
        img_mod = self.data.loc[idx, 'mod']
        img = Image.open(file_path).convert('RGB')
        img = self._apply_mod(img, img_mod)

        if self.transform:
            img = self.transform(img)

        return img, label
        

    def _apply_mod(self, img : Image.Image, img_mod : str, noise_scale : float = 0.02, apply_noise : bool = True) -> Image.Image:
        '''
        Helper function to apply the modificaation.

        Parameters
        ----------
        img : Image.Image
            image to apply the modification to
        img_mod : str
            type of modification to apply
        noise_scale : float
            amount of noise to appy to any time
        apply_noise : bool
            do or do not apply the noise

        Returns
        -------
        img_w_mod : Image.Image
            image with the modification applied
        ''' 
        if img_mod == 'rot1':
            img = img.rotate(90)
        elif img_mod == 'rot2':
            img = img.rotate(180)
        elif img_mod == 'rot3':
            img = img.rotate(270)
        elif img_mod == 'mir_h':
            img = ImageOps.flip(img)
        elif img_mod == 'mir_v':
            img = ImageOps.mirror(img)
        elif img_mod == 'transpose':
            img = Image.fromarray(np.transpose(np.array(img), (1,0,2)))  # swap H/W; (0,1,2) was a no-op (P1-18)

        img_arr = np.array(img)
        img_shape = img_arr.shape
        noise = np.zeros(img_shape)

        if apply_noise:    
            noise_vals = np.random.normal(scale=noise_scale, size=img_shape)*255
            noise = noise_vals
            
        img_w_noise = img_arr + np.round(noise).astype(int)
        img_w_noise[img_w_noise > 255] = 255
        img_w_noise[img_w_noise < 0] = 0
    
        r_img = Image.fromarray(img_w_noise.astype(np.uint8))
        #r_img = r_img.resize(self.new_size) #=(299,299)
        return r_img
    
#from .segmentation_2 import UNet
import matplotlib.pyplot as plt
import torch

class ISICDatasetSegmentation(Dataset):
    '''
    DataLoader that will use a segmenation model to first
    perform image segmentation as a preprocessing step.

    Attributes
    ----------
    df : pd.DataFrame
        training data layed out in a csv. Must have columns
        'image_path', 'target', 'image_id', 'mod'
    size : tuple[int, int]
        size to reshape images to as required by the mdoel
    transform : torchvision.transforms.Compose
        traning augmentations to aid in training
    seg_kwargs : dict
        information pertaining to segmenation.
    '''
    def __init__(self, df: pd.DataFrame, size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None, seg_kwargs : Optional[dict] = None):

        if seg_kwargs is None:
            raise ValueError('ISICDatasetSegmentation requires seg_kwargs with "seg_model", "seg_model_path" and "seg_model_device".')
        super().__init__()
        self.data = df
        self.size = size
        self.transform=transform
        self.seg_model = self._get_seg_model(seg_kwargs)
        self.seg_model_path = self._get_seg_model_path(seg_kwargs)
        self.seg_model_device = self._get_seg_model_device(seg_kwargs)

        self.seg_model.load_state_dict(torch.load(self.seg_model_path)['state_dict']) 
        self.seg_model.eval()
        self.seg_model.to(self.seg_model_device)
   
    def _get_seg_model(self, seg_kwargs : dict):
        '''
        Get the model from the seg_kwargs dict, but first validate
        it is present.

        Parameters
        ----------
        seg_kwargs : dict
            data pertaining to segmentation

        Returns
        -------
        seg_model_class : uknown
            pytorch segmantion model class
        '''

        if 'seg_model' in seg_kwargs.keys():
            return seg_kwargs['seg_model']
        else:
            raise KeyError('Tried getting segmentation model from "seg_kwargs", but "seg_model" was not a key')

    def _get_seg_model_device(self, seg_kwargs : dict) -> str:
        '''
        Get the model device from the seg_kwargs dict, but first validate
        it is present. Must be 'cpu' or 'cuda'

        Parameters
        ----------
        seg_kwargs : dict
            data pertaining to segmentation

        Returns
        -------
        device : str
            device to map model to
        '''
        if 'seg_model_device' in seg_kwargs.keys():
            return seg_kwargs['seg_model_device']
        else:
            raise KeyError('Tried getting segmentation runtime device from "seg_kwargs", but "seg_model_device" was not a key.')
    
    def _get_seg_model_path(self, seg_kwargs : dict) -> str:
        '''
        Get the model save path from the seg_kwargs dict, but first validate
        it is present. 

        Parameters
        ----------
        seg_kwargs : dict
            data pertaining to segmentation

        Returns
        -------
        path : str
            path to frozen segmentation model 
        '''
        if 'seg_model_path' in seg_kwargs.keys():
            return seg_kwargs['seg_model_path']
        else:
            raise KeyError('Tried getting path to saved segmentation model weights from "seg_kwargs", but "seg_model_path" was not a key.')
    
    def __len__(self):

        return len(self.data)
    

    def __getitem__(self, idx):
    
        # Same preprocessing the UNet was trained with (RGB / 255, 256x256); see archive/development/segmentation_2.py
        img = np.asarray(Image.open(self.data.loc[idx, 'image_path']).convert('RGB'), dtype=np.float32) / 255.0
        lbl = self.data.loc[idx, 'target']

        orig_h, orig_w = img.shape[:2]
        img_256 = cv2.resize(img, (256, 256))

        with torch.no_grad():
            img_tensor = torch.from_numpy(img_256).permute(2, 0, 1).unsqueeze(0).to(self.seg_model_device)
            # UNet was trained with BCEWithLogitsLoss: threshold sigmoid(logit) at 0.5, as in its own evaluation.
            # (Previously the raw logits were min-max normalised per image, which always produced a "lesion".)
            prob_mask = torch.sigmoid(self.seg_model(img_tensor)).squeeze().cpu().numpy()

        prob_mask = cv2.resize(prob_mask, (orig_w, orig_h))  # cv2 takes (width, height)
        prob_mask = cv2.GaussianBlur(prob_mask, (11, 11), 0)
        binary_mask = (prob_mask > 0.5).astype(np.float32)

        model_img = img * binary_mask[..., None]
        model_img = Image.fromarray((model_img * 255).astype(np.uint8))

        if self.transform:
            model_img = self.transform(model_img)

        return model_img, lbl
    
    
class ISICMixupDataset(Dataset):
    
    '''
    Data loader that used the DataMixup protocol to 
    generate training examples

    Attributes
    ----------
    df : pd.DataFrame
        training data layed out in a csv. Must have columns
        'image_path', 'target', 'image_id', 'mod'
    img_size : tuple[int, int]
        size to reshape images to as required by the mdoel
    transform : torchvision.transforms.Compose
        traning augmentations to aid in training
    seg_kwargs : dict
        information pertaining to segmenation.
    '''

    def __init__(self, df: pd.DataFrame, img_size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None, seg_kwargs : Optional[dict] = None, blur_radius : float = 0):

        super().__init__()
        self.blur_radius = blur_radius  # was always 3; inference never blurs, so off by default (P0-7)
        self.img_size = img_size
        self.data = df.reset_index(drop=True)
        self.transform=transform
        self.required_cols = set(['image_path_1', 'image_path_2','target','lambda','image_id_1','image_id_2'])
        
    def _validate_cols(self) -> None:
        '''
        Validate the manditory columns are present
        ''' 
        df_cols = set(self.data.columns)
        col_diff = list(self.required_cols.difference(df_cols))

        if len(col_diff) > 0:
            col_diff = ['"'+str(c)+'"' for c in col_diff]
            raise AttributeError(f'Reference DataFrame has incorrect columns. Columns: {','.join([c for c in col_diff])} are Missing.')
 
    def __len__(self)-> int:

        return len(self.data)

    def __getitem__(self, idx):

        file_path_1 = self.data.loc[idx, 'image_path_1']
        file_path_2 = self.data.loc[idx, 'image_path_2']
        lam = self.data.loc[idx, 'lambda']
        label = self.data.loc[idx, 'target']
        
        img_1 = Image.open(file_path_1).convert('RGB').resize(self.img_size)
        img_2 = Image.open(file_path_2).convert('RGB').resize(self.img_size)

        img = lam*np.array(img_1) + (1-lam)*np.array(img_2)
        img = Image.fromarray(img.astype(np.uint8))
        if self.blur_radius > 0:
            img = self.gauss_blur(img)
        
        if self.transform:
            img = self.transform(img)

        return img, label
    

    def gauss_blur(self, img : Image.Image):
        '''
        Apply Gaussian blur to an image

        Parameters
        ----------
        img : Image.Image
            image to apply the blur to

        Returns
        -------
        img : Image.Image
            image with blur
        '''
        return img.filter(ImageFilter.GaussianBlur(radius = self.blur_radius))


from torchvision.transforms.functional import pil_to_tensor
class ISICFeatureFusionMixupDataset(Dataset):
    '''
    Data loader that used the DataMixup protocol to 
    generate training examples for feature fustion.
    As difference models have different input size
    special modification are required to keep training
    data consistent. 

    Attributes
    ----------
    df : pd.DataFrame
        training data layed out in a csv. Must have columns
        'image_path', 'target', 'image_id', 'mod'
    img_size_1 : tuple[int, int]
        size to reshape images to as required by the mdoel
    image_size_2 : tuple[int, int]
        second size to reshape the image
    transform : torchvision.transforms.Compose
        traning augmentations to aid in training
    seg_kwargs : dict
        information pertaining to segmenation.
    '''
    def __init__(self, df: pd.DataFrame, img_size, img_size_2=None, transform=None, as_tensor=None, blur_radius : float = 0):

        super().__init__()
        self.blur_radius = blur_radius  # previously always on; off by default to match inference (P0-7)
        self.img_size = img_size
        self.img_size_2 = img_size_2
        self.data = df.reset_index(drop=True)
        self.transform=transform
        self.as_tensor=as_tensor
        print(len(df))
        

    def __len__(self)-> int:

        return len(self.data)


    def __getitem__(self, idx):

        file_path_1 = self.data.loc[idx, 'image_path_1']
        file_path_2 = self.data.loc[idx, 'image_path_2']
        lam = self.data.loc[idx, 'lambda']
        label = self.data.loc[idx, 'target']
        
        img_1 = Image.open(file_path_1).convert('RGB').resize(self.img_size)
        img_2 = Image.open(file_path_2).convert('RGB').resize(self.img_size)

        img = lam*np.array(img_1) + (1-lam)*np.array(img_2)
        img = Image.fromarray(img.astype(np.uint8))
        if self.blur_radius > 0:
            img = self.gauss_blur(img)
        
        if self.transform:
            img = self.transform(img)

        if self.img_size_2:
            img_2 = copy.deepcopy(img).resize(self.img_size_2)
        
        if self.as_tensor:
            img = self.as_tensor(img)
            img_2 = self.as_tensor(img_2)

        if self.img_size_2:
            return img, img_2, label
        else:
            return img, label
    

    def gauss_blur(self, img: Image.Image) -> Image.Image:
        '''
        Apply Gaussian blur to an image

        Parameters
        ----------
        img : Image.Image
            image to apply the blur to

        Returns
        -------
        img : Image.Image
            image with blur
        '''
        return img.filter(ImageFilter.GaussianBlur(radius = self.blur_radius))


