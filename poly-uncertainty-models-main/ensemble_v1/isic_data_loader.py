import cv2
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from albumentations.core.transforms_interface import ImageOnlyTransform


class ISICDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform=None):

        super().__init__()
        self.data = df.reset_index(drop=True)
        self.transform=transform
        

    def __len__(self)-> int:

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
    

    def gauss_blur(self, img):

        return img.filter(ImageFilter.GaussianBlur(radius = 9))


class ISICDatasetWMod(Dataset):

    def __init__(self, df:pd.DataFrame, transform=None):

        super().__init__()
        self.data = df
        self.transform=transform

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
        

    def _apply_mod(self, img, img_mod, noise_scale=0.02, apply_noise=True):
    
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
    
from unet_model import UNet
import matplotlib.pyplot as plt
import torch

class ISICDatasetSegmentation(Dataset):

    def __init__(self, df:pd.DataFrame, segmentation_model_path:str = './seg_model_checkpoint_last.pth.tar', transform=None):

        super().__init__()
        self.data = df
        self.transform=transform
        self.seg_model = UNet(n_channels=3, n_classes=1).to(DEVICE)
        self.seg_model.load_state_dict(torch.load(segmentation_model_path)['state_dict']) 
        self.seg_model.to('cpu')


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
    


class ISICDatasetPreloaded(Dataset):

    def __init__(self, df:pd.DataFrame, new_size, transform=None):

        super().__init__()
        self.data = df
        self.size = new_size
        self.img_arr, self.img_lbl = self._preload_data()
        self.transform=transform

    def __len__(self):

        return len(self.img_arr)

    def __getitem__(self, idx):
        img, lbl = self.img_arr[idx], self.img_lbl[idx]

        if self.transform:
            img = self.transform(img)

        return self.img_arr


    def _apply_mod(self, img, img_mod, noise_scale=0.02, apply_noise=True):
    
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
        r_img = r_img.resize(self.size) #=(299,299)
        return r_img
    

    def _preload_data(self):

        #mod = ['None','rot1','rot2','rot3','mir_h','mir_v','transpose']
        w, h = self.size
        train_arr = np.zeros((len(self.data), w, h, 3))
        train_lbl = np.zeros((len(self.data)))

        print('Preloading all training data...')
        for i, row in self.data.iterrows():

            if i%1000 == 0:
                print(f'Completed processing {i}/{len(self.data)} images')
            p = row['image_path']
            l = row['target']
            m = row['mod']
            img = Image.open(p)
            img_w_mod = self._apply_mod(img, m, noise_scale=0.02, apply_noise=True) 
            train_arr[i] = np.array(img_w_mod)
            train_lbl[i] = l

        return train_arr, train_lbl
                
    
class ISICMixupDataset(Dataset):

    def __init__(self, df: pd.DataFrame, img_size, transform=None):

        super().__init__()
        self.img_size = img_size
        self.data = df.reset_index(drop=True)
        self.transform=transform
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

        return img, label
    

    def gauss_blur(self, img):

        return img.filter(ImageFilter.GaussianBlur(radius = 3))


from torchvision.transforms.functional import pil_to_tensor
class ISICFeatureFusionMixupDataset(Dataset):

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
    

    def gauss_blur(self, img):

        return img.filter(ImageFilter.GaussianBlur(radius = 3))


