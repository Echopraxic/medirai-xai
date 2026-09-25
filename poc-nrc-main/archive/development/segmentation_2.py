import cv2
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import albumentations as A

import os
import matplotlib.pyplot as plt

import glob
import pandas as pd

import copy

IMG_PATH = '../../data/ISIC2018_Task1-2_Training_Input/'
NO_MASK_PATH = '../../data/skin_texture_pruned/'
MASK_PATH = '../../data/ISIC2018_Task1_Training_GroundTruth/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE= 32
SIZE = 256
LEARNING_RATE = 0.0003
NUM_EPOCHS = 30
NUM_WORKERS = 2
PIN_MEMORY = True

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path (encoder)
        self.conv1 = nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path (decoder)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        # Contracting path (encoder)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))
        x5 = F.relu(self.conv5(self.pool(x4)))

        # Expansive path (decoder)
        x6 = F.relu(self.upconv1(x5))
        x6 = torch.cat([x4, x6], dim=1)
        x6 = F.relu(self.conv6(x6))
        x7 = F.relu(self.upconv2(x6))
        x7 = torch.cat([x3, x7], dim=1)
        x7 = F.relu(self.conv7(x7))
        x8 = F.relu(self.upconv3(x7))
        x8 = torch.cat([x2, x8], dim=1)
        x8 = F.relu(self.conv8(x8))
        x9 = F.relu(self.upconv4(x8))
        x9 = torch.cat([x1, x9], dim=1)
        x9 = F.relu(self.conv9(x9))
        x10 = self.conv10(x9)

        return x10
    

class ISICDataset(Dataset):

    def __init__(self, images_path, masks_path, size, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.ids = [image_file[:-4] for image_file in os.listdir(images_path) if image_file.endswith('.jpg')]
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.ids[idx] + '.jpg')
        mask_path = os.path.join(self.masks_path, self.ids[idx] + '_segmentation.png')
        
        # Load image and mask
        img = cv2.imread(os.path.join(self.images_path, self.ids[idx] + '.jpg'), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.masks_path, self.ids[idx] + '_segmentation.png'), cv2.IMREAD_GRAYSCALE)
        
         # Convert to RGB, And convert mask to binary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations['image']
            mask = augmentations['mask']
        
        # Convert numpy arrays to PyTorch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        mask[mask == 255.0] = 1.0

        return img, mask
    
class ExpandedISICDataset(Dataset):

    def __init__(self, df, size, transform=None):
        self.df = df
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'image_path']
        mask_path = self.df.loc[idx, 'mask_path']

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        if mask_path != '-1':
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            #print(image_path, mask_path)
            #print(mask.shape, img.shape)

        else:
            mask = np.zeros(img.shape[:2])
            
        
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations['image']
            mask = augmentations['mask']
        
        # Convert numpy arrays to PyTorch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        mask[mask == 255.0] = 1.0

        #print(img.shape, mask.shape)
        return img, mask

'''
#create training dataframe
images = sorted(glob.glob(IMG_PATH+'*.jpg'))# = '../../data/ISIC2018_Task1-2_Training_Input/'
no_mask_images = glob.glob(NO_MASK_PATH+'*.tif')# = '../../data/skin_texture_pruned/'
masks = sorted(glob.glob(MASK_PATH+'*.png'))# = '../../data/ISIC2018_Task1_Training_GroundTruth/'
no_mask_masks = ['-1']*len(no_mask_images)
#print(len(images), len(no_mask_images))

for i in range(10):
    print(images[i])
    print(masks[i])
    print()

data_df = pd.DataFrame()
data_df['image_path'] = images+no_mask_images
data_df['mask_path'] = masks+no_mask_masks
data_df = data_df.sample(frac=1).reset_index(drop=True)
print(data_df.head())

split_idx = int(0.8 * len(data_df)) 
train_df, test_df = data_df[:split_idx], data_df[split_idx:]
train_df.to_csv('../../results/segmentation/train_df.csv', index=False)
test_df.to_csv('../../results/segmentation/test_df.csv', index=False)
'''
transform = A.Compose(
[
    A.Resize(height=SIZE, width=SIZE),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1)
])

#dataset = ISICDataset(images_path=IMG_PATH,
#                      masks_path=MASK_PATH, 
#                      size=SIZE, 
#                      transform=transform)

#dataset = ExpandedISICDataset(
#                train_df,
#                size=SIZE,
#                transform=transform
#            )


#train_size = int(0.8 * len(dataset))
#train_dataset = Subset(dataset, range(train_size))
#test_dataset = Subset(dataset, range(train_size, len(dataset)))
#print(test_dataset)

train_df = pd.read_csv('../../results/segmentation/train_df.csv')#, index=False)
test_df = pd.read_csv('../../results/segmentation/test_df.csv')#, index=False)

train_dataset = ExpandedISICDataset(train_df, size=SIZE, transform=transform)
test_dataset = ExpandedISICDataset(test_df, size=SIZE, transform=transform) 


train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          #num_workers=NUM_WORKERS, 
                          #pin_memory=PIN_MEMORY
                          )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

img, mask = test_dataset[18]
img = img.permute(1, 2, 0).numpy()
mask = mask.permute(1, 2, 0).squeeze().numpy()
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
mask = np.expand_dims(mask, axis=2)
mask = np.repeat(mask, 3, axis=2)
concatenated_img = np.concatenate((img, mask), axis=1)

fig, ax = plt.subplots()
ax.imshow(concatenated_img)
ax.set_title('Image', loc='left')
ax.set_title('Mask', loc='right')

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.show()

model = UNet(n_channels=3, n_classes=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    dice_score = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8
            
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

'''
for epoch in range(NUM_EPOCHS): 
    
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    
    if epoch % 5 == 0 and epoch != 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, "../../saved_models/segmentation_2/checkpointN"+str(epoch)+"_.pth.tar")  
    
    
    for batch_idx, (images, masks) in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        # Gradients to 0
        optimizer.zero_grad()

        # Forward 
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_description(f"Epoch[{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss = loss.item())


checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
torch.save(checkpoint, "../../saved_models/segmentation_w_blanks/checkpoint_last.pth.tar")
'''

model.load_state_dict(torch.load('../../saved_models/segmentation_w_blanks/checkpoint_last.pth.tar')['state_dict'])
#check_accuracy(test_loader, model, DEVICE)

# Load an example image and mask

test_data = pd.read_csv('./test_EQ.csv')

#from medirai_models_w_hidden import MediraiDensenetHiddenModel, MediraiEfficientNetHiddenModel, MediraiInceptionHiddenModel, MediraiResNetHiddenModel
from medirai_base_model import MediraiEnsembleModelV1 
model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)
from sklearn.metrics import confusion_matrix

print('Loading Pretained Models...')    
#model_with_hidden.load(path_to_densenet='../../saved_models/medirai_models_with_hidden/dense_net_w_hid_256_best.pkl')
#model_with_hidden.load(path_to_efficientnet='../../saved_models/medirai_models_with_hidden/efficient_net_w_hid_256_best.pkl')
#model_with_hidden.load(path_to_inception='../../saved_models/medirai_models_with_hidden/inception_w_hid_256_best.pkl')
model_with_hidden.load(path_to_resnet='../../saved_models/medirai_models_seg/res_net_w_hid_5.pkl')
print(model_with_hidden)

preds, truth = [], []

for i, row in test_data.iterrows():

    img = plt.imread(row['image_path'])/255.0
    img_orig = copy.deepcopy(img)
    img = resized_image = cv2.resize(img, (SIZE, SIZE)) 
    #img = torch.from_numpy(img).permute(1, 2, 0).float() / 255. 
    #plt.show()

    with torch.no_grad():
        img_tensor = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
        generated_mask = model(img_tensor).squeeze().cpu().numpy()

    generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
    generated_mask_resized = cv2.GaussianBlur(generated_mask_resized,(11,11),0)
    generated_mask_resized = (generated_mask_resized-np.min(generated_mask_resized))/(np.max(generated_mask_resized)-np.min(generated_mask_resized))
    generated_mask_stacked = np.stack((generated_mask_resized,)*3, axis=-1)
    generated_mask_stacked = (generated_mask_stacked > 0.5).astype(int)

    img = (255*img)*(generated_mask_stacked)
    img = img.astype(np.uint8)
    #print(img)
    pred = model_with_hidden.resnet.predict(img, 'cpu')
    true = row['target']

    preds.append(pred[1][1])
    truth.append(true)

    #print(pred, true)
    
    fig, ax = plt.subplots(1, 3, figsize=(8, 5))

    #img = torch.from_numpy(img).permute(1, 2, 0).float() / 255. 
    #img = img.permute(1, 2, 0).numpy() 
    ax[0].imshow(img_orig)
    ax[0].set_title('Original Image')

    ax[1].imshow(generated_mask_stacked*255)
    ax[1].set_title('Generated Mask')

    cv2.imwrite(f'../../results/seg_masks/masks_{row['image_id']}.png', generated_mask_stacked*255)

    ax[2].imshow(img)
    ax[2].set_title('Segmented Image')

    #ax[2].imshow(d)
    plt.savefig(f'../../results/segmentation_on_test/image_masks_{i}_({pred[1][1]}, {true}).png')
    

trues = np.array(truth)
preds = (np.array(preds) > 0.5).astype(int)

print(confusion_matrix(trues, preds))

assert 1 == 0

for i in range(len(test_dataset)):
    img, true_mask = test_dataset[i]
    img = img.permute(1, 2, 0).numpy()
    true_mask = true_mask.permute(1, 2, 0).squeeze().numpy()

    # Generate a predicted mask using your trained U-net model
    with torch.no_grad():
        img_tensor = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
        generated_mask = model(img_tensor).squeeze().cpu().numpy()

    # Resize and stack the masks for display
    true_mask_resized = cv2.resize(true_mask, (img.shape[1], img.shape[0]))
    true_mask_stacked = np.stack((true_mask_resized,)*3, axis=-1)

    generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
    generated_mask_stacked = np.stack((generated_mask_resized,)*3, axis=-1)

    # Combine the image and masks into a single plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img)
    ax[0].set_title('Original Image')

    ax[1].imshow(true_mask_stacked)
    ax[1].set_title('True Mask')

    ax[2].imshow(generated_mask_stacked)
    ax[2].set_title('Generated Mask')
    plt.savefig(f'../../results/segmentation/image_masks_{i}.png')
    #plt.show()

