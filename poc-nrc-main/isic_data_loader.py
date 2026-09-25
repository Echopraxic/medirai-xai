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

    def __init__(self, df: pd.DataFrame, size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None):

        super().__init__()
        self.data = df.reset_index(drop=True)
        self.size = size
        self.transform=transform
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

        if 'melanoma' in image_id:
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

        return img.filter(ImageFilter.GaussianBlur(radius = 9))


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
            img = Image.fromarray(np.transpose(np.array(img), (0,1,2)))

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
    def __init__(self, df: pd.DataFrame, size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None, seg_kwargs : dict):
    #def __init__(self, df:pd.DataFrame, size : tuple[int, int], transform=None, seg_kwargs=None):
                         
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
    
        img = plt.imread(self.data.loc[idx, 'image_path'])/255.0
        lbl = self.data.loc[idx, 'target']

        img_size = img.shape
        img = cv2.resize(img, (256, 256))
       
        with torch.no_grad():
            img_tensor = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to('cpu')
            generated_mask = self.seg_model(img_tensor).squeeze().cpu().numpy()

        generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
        generated_mask_resized = cv2.GaussianBlur(generated_mask_resized,(11,11),0)
        generated_mask_resized = (generated_mask_resized-np.min(generated_mask_resized))/(np.max(generated_mask_resized)-np.min(generated_mask_resized))
        generated_mask_stacked = np.stack((generated_mask_resized,)*3, axis=-1)
        generated_mask_stacked = (generated_mask_stacked > 0.5).astype(int)

        model_img = img*generated_mask_stacked
        model_img = cv2.resize(model_img, (img_size[0], img_size[1]))

        if self.transform:
            train_img = self.transform(train_img)

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

    #def __init__(self, df: pd.DataFrame, img_size, transform=None):
    def __init__(self, df: pd.DataFrame, img_size : tuple[int, int], transform: Optional[torchvision.transforms.Compose] = None, seg_kwargs : dict):

        super().__init__()
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
        return img.filter(ImageFilter.GaussianBlur(radius = 3))


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
    def __init__(self, df: pd.DataFrame, img_size, img_size_2=None, transform=None, as_tensor=None):

        super().__init__()
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
        return img.filter(ImageFilter.GaussianBlur(radius = 3))


