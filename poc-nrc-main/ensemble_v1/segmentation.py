
import glob
import pandas as pd
import numpy as np

from PIL import Image, ImageFilter
from torch import nn, optim

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

data_dir = '../../data/ISIC2018_Task1-2_Training_Input/'
true_dir = '../../data/ISIC2018_Task1_Training_GroundTruth/'

data = {}
imgs_paths = glob.glob(data_dir+'*.jpg')
mask_paths = glob.glob(true_dir+'*.png')

print('Testing:')
print(len(imgs_paths))
print(len(mask_paths))

data_df = []
for ip in imgs_paths:

    obs_id = ip.split('/')[-1].split('.')[0]
    related_masks = [mp for mp in mask_paths if obs_id in mp]

    if len(related_masks) > 1:
        continue
    
    data_df.append({
        'id':obs_id,
        'image_path': ip,
        'mask_path': related_masks[0],
    })

data_df = pd.DataFrame(data_df)
print(data_df.head())

def data_transforms():

    transform={
    'image': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'mask':transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        ])
    }
    return transform


from torch.utils.data import DataLoader, Dataset
class ISICSegmentation(Dataset):

    def __init__(self, df: pd.DataFrame, transforms):

        super().__init__()
        self.data = df.reset_index(drop=True)
        self.transforms = transforms
        

    def __len__(self)-> int:

        return len(self.data)


    def __getitem__(self, idx):

        file_path = self.data.loc[idx, 'image_path']
        mask_path = self.data.loc[idx, 'mask_path']

        img = Image.open(file_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        img = self.transforms['image'](img)
        mask = self.transforms['mask'](mask)

        #print(img.shape, mask.shape)
        #print(torch.max(mask))
        return img, mask 
    
    def gauss_blur(self, img):

        return img.filter(ImageFilter.GaussianBlur(radius = 9))


train_dataset = ISICSegmentation(data_df, data_transforms())
print(len(train_dataset))
train_loader = DataLoader(
                train_dataset,
                batch_size=16,
                shuffle=True,
                drop_last=True,
                #num_workers=1,
            )

import segmentation_models_pytorch as smp

loss_func = nn.CrossEntropyLoss()

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
model.to('cpu')

optimizer = optim.AdamW(
                            model.parameters(), 
                            lr=0.0001, 
                            weight_decay=0.5e-4,
                        )

#print('Training...')
#print(len(train_loader))

'''
num_epochs = 5
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}')
    for image, gt_masks in train_loader:

        #print(image, gt_masks)
        predicted_mask = model(image)
        loss = loss_func(predicted_mask, gt_masks)

        loss.backward()
        optimizer.step()

model.save_pretrained('../../saved_models/segmentation')
'''

img_transforms = data_transforms() 
sample_img = Image.open('./sample_img.jpg').convert('RGB')


model = smp.from_pretrained('../../saved_models/segmentation')
mask = model(img_transforms['image'](sample_img).unsqueeze(0))
pred_mask = (mask.detach().numpy()[0] > 0.5).astype(int)

import matplotlib.pyplot as plt
plt.imshow(pred_mask[0])
plt.show()
