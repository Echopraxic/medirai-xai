
from ensemble_v1.medirai_base_model import MediraiEnsembleModelV1
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

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

'''
model = UNet(n_channels=3, n_classes=1).to('cpu')
model.load_state_dict(torch.load('../../saved_models/segmentation_w_blanks/checkpoint_last.pth.tar')['state_dict'])

model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)

#model_with_hidden.load(path_to_densenet='../../saved_models/medirai_models_with_hidden/dense_net_w_hid_256_best.pkl')
#model_with_hidden.load(path_to_efficientnet='../../saved_models/medirai_models_with_hidden/efficient_net_w_hid_256_best.pkl')
#model_with_hidden.load(path_to_inception='../../saved_models/medirai_models_with_hidden/inception_w_hid_256_best.pkl')
model_with_hidden.load(path_to_resnet='../../saved_models/data_preprocess_exp/res_net_w_hid_best_mixup.pkl')
#model_with_hidden.load(path_to_resnet='../../saved_models/data_preprocess_exp/res_net_w_hid_seg_convex.pkl')#res_net_w_hid_4.pkl
#model_with_hidden.load(path_to_resnet='../../saved_models/data_preprocess_exp/res_net_w_hid_best_seg.pkl')
is_seg = False
is_convex = False 

test_data = pd.read_csv('./test_EQ.csv')
SIZE = 256
'''


def thresh_callback(src_gray, thresh=100):

        canny_output = cv2.Canny(src_gray, thresh, thresh * 2)

        contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found.")
            return src_gray

        # Find the outermost contour
        def bounding_area(c):
            x, y, w, h = cv2.boundingRect(c)
            return w * h

        outermost_contour = max(contours, key=bounding_area)
        hull = cv2.convexHull(outermost_contour)

        drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(drawing, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)

        return drawing

'''
img_1 = plt.imread(test_data.loc[0, 'image_path'])
img_2 = plt.imread(test_data.loc[2, 'image_path'])[:125,:125,:]

img_n = img_1*(0.5)+(1-0.5)*img_2
img_n = img_n.astype(np.uint8)

fig, ax = plt.subplots(1, 3, figsize=(8, 5))

ax[0].imshow(img_1)
ax[0].set_title('Image 1')

ax[1].imshow(img_2)
ax[1].set_title('Image 2')

ax[2].imshow(img_n)
ax[2].set_title('Data Mixup')
plt.show()
'''

'''
for k in range(5):

    model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)

    #model_with_hidden.load(path_to_densenet='../../saved_models/medirai_models_with_hidden/dense_net_w_hid_256_best.pkl')
    #model_with_hidden.load(path_to_efficientnet='../../saved_models/medirai_models_with_hidden/efficient_net_w_hid_256_best.pkl')
    #model_with_hidden.load(path_to_inception='../../saved_models/medirai_models_with_hidden/inception_w_hid_256_best.pkl')
    #model_with_hidden.load(path_to_resnet='../../saved_models/data_preprocess_exp/res_net_w_hid_best_mixup.pkl')
    #model_with_hidden.load(path_to_resnet='../../saved_models/data_preprocess_exp/res_net_w_hid_seg_convex.pkl')#res_net_w_hid_4.pkl
    #model_with_hidden.load(path_to_resnet='../../saved_models/data_preprocess_exp/res_net_w_hid_best_seg.pkl')

    model_with_hidden.load(path_to_densenet=f'../../saved_models/dnn_models_mixup/run1/dense_net_w_hid_{k+1}.pkl')
    model_with_hidden.load(path_to_efficientnet=f'../../saved_models/dnn_models_mixup/run1/efficient_net_w_hid_{k+1}.pkl')
    model_with_hidden.load(path_to_inception=f'../../saved_models/dnn_models_mixup/run1/inception_w_hid_{k+1}.pkl')
    model_with_hidden.load(path_to_resnet=f'../../saved_models/dnn_models_mixup/run1/res_net_w_hid_{k+1}.pkl')

    is_seg = False
    is_convex = False 

    preds = []
    truth = []

    for i in range(len(test_data)):

        img_p = test_data.loc[i, 'image_path']
        img = plt.imread(img_p)

        if is_seg:
            img = img/255.0
            img_orig = copy.deepcopy(img)
            img = resized_image = cv2.resize(img, (SIZE, SIZE)) 
    a        #img = torch.from_numpy(img).permute(1, 2, 0).float() / 255. 
            #plt.show()

            with torch.no_grad():
                img_tensor = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to('cpu')
                generated_mask = model(img_tensor).squeeze().cpu().numpy()

            generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
            generated_mask_resized = cv2.GaussianBlur(generated_mask_resized,(11,11),0)
            generated_mask_resized = (generated_mask_resized-np.min(generated_mask_resized))/(np.max(generated_mask_resized)-np.min(generated_mask_resized))
            generated_mask_stacked = np.stack((generated_mask_resized,)*3, axis=-1)
            generated_mask_stacked = (generated_mask_stacked > 0.5).astype(int)
            orig_mask = copy.deepcopy(generated_mask_stacked)

            if is_convex:

                cleaned_mask = thresh_callback((255*generated_mask_stacked[:,:,0]).astype(np.uint8))
                cleaned_mask = cv2.cvtColor(cleaned_mask, cv2.COLOR_BGR2RGB)
            
                generated_mask_stacked = cleaned_mask/255.0
            
            model_img = (255*img)*generated_mask_stacked

            img = (255*img)*(generated_mask_stacked)
            img = img.astype(np.uint8)

        #fig, ax = plt.subplots(1, 3, figsize=(8, 5))

        #ax[0].imshow(img_orig)
        #ax[0].set_title('Original Image')

        #ax[1].imshow(orig_mask*img)
        #ax[1].set_title('Segmentation')

        #ax[2].imshow(img)
        #ax[2].set_title('Convex Hull')
        #plt.show()
        
        #plt.imshow(img)
        #plt.show()       
        d_pred = model_with_hidden.densenet.predict(img_p, 'cpu')
        e_pred = model_with_hidden.efficientnet.predict(img_p, 'cpu')
        i_pred = model_with_hidden.inception.predict(img_p, 'cpu')
        r_pred = model_with_hidden.resnet.predict(img_p, 'cpu')

        pred = np.mean([d_pred[1][1], e_pred[1][1], i_pred[1][1], r_pred[1][1]])

        #print(pred[1][1], '-', test_data.loc[i, 'target'])
        preds.append(pred)#pred[1][1])
        truth.append(test_data.loc[i, 'target'])

    preds = (np.array(preds) > 0.5).astype(int)
    truth = np.array(truth)
    
    print('=========================')
    print(len(preds), len(truth))
    print(f'Fold {k+1}:')
    print(confusion_matrix(truth, preds))
    print('=========================')
'''    

#=========================
'''
k=4

from feature_fusion_mixup import FeatureFusionMixup
from feature_fusion import FeatureFusion
print('Feature Fusion:')

ff_model = FeatureFusion(
            #dnn_hid=32,
            #prednet_hid=64
        )

dnn_paths = [
    #f'../../saved_models/dnn_models_mixup/run1/dense_net_w_hid_best.pkl',
    #f'../../saved_models/dnn_models_mixup/run1/efficient_net_w_hid_best.pkl',
    #f'../../saved_models/dnn_models_mixup/run1/inception_w_hid_best.pkl',
    f'../../saved_models/2025-06-12/dense_net_w_hid_best_mixup.pkl',
    f'../../saved_models/2025-06-12/efficient_net_w_hid_best_mixup.pkl',
    f'../../saved_models/2025-06-12/inception_w_hid_best_mixup.pkl',
    f'../../saved_models/2025-06-12/res_net_w_hid_best_mixup.pkl',
        ]

ff_nn_paths = [
            f'../../saved_models/dnn_new_negs/dense_net_w_hid_best_mixup_new_negs.pkl',
            f'../../saved_models/dnn_new_negs/efficient_net_w_hid_best_mixup_new_negs.pkl',
            f'../../saved_models/dnn_new_negs/inception_w_hid_best_mixup_new_negs.pkl',
            f'../../saved_models/dnn_new_negs/res_net_w_hid_best_mixup_new_negs.pkl',
        ]

ff_nn_paths = [
            f'../../saved_models/dnns_sigmoid/dense_net_w_hid_best_mixup_new_negs_sigmoid.pkl',
            f'../../saved_models/dnns_sigmoid/efficient_net_w_hid_best_mixup_new_negs_sigmoid.pkl',
            f'../../saved_models/dnns_sigmoid/inception_w_hid_best_mixup_new_negs_sigmoid.pkl',
            f'../../saved_models/dnns_sigmoid/res_net_w_hid_best_mixup_new_negs_sigmoid.pkl',
        ]
    
ff_nn_paths = [
            f'../../saved_models/medirai_models_with_hidden/dense_net_w_hid_256_best.pkl',
            f'../../saved_models/medirai_models_with_hidden/efficient_net_w_hid_256_best.pkl',
            f'../../saved_models/medirai_models_with_hidden/inception_w_hid_256_best.pkl',
            f'../../saved_models/medirai_models_with_hidden/res_net_w_hid_256_best.pkl',
        ]
    
ff_model.load_dnn(ff_nn_paths)

for m in range(0, 20):

    ff_model.load_pred_net(f'../../saved_models/ff_temp/feature_fusion_epoch-{m}-og.plk')

    preds = []
    truth = []

    for i in range(len(test_data)):

        img_p = test_data.loc[i, 'image_path']
        #img = plt.imread(img_p)

        pred = ff_model.predict(img_p)
        #print(pred)

        #print(pred[1][1], '-', test_data.loc[i, 'target'])
        preds.append(pred[1][1])
        truth.append(test_data.loc[i, 'target'])

    preds = (np.array(preds) > 0.5).astype(int)
    truth = np.array(truth)

    print('=========================')
    print(f'Model epoch {m}')
    print(confusion_matrix(truth, preds))
    print('=========================')

print()
'''
import os

def show_image(f_path):
    
   #f_path = f'./train/{id}.jpg'
   img = plt.imread(f_path)
   plt.imshow(img)
   plt.show()


test_data = pd.read_csv('./ensemble_v1/test_EQ.csv')
img_paths = test_data['image_path']
corrected_paths = [p.replace('../../../', '../../') for p in img_paths]
test_data['image_path'] = corrected_paths

dnn_paths = [
    f'../saved_models/dnns_mixup_1M_batchnorm/dense_net_w_hid_best_mixup_1M.pkl',
    f'../saved_models/dnns_mixup_1M_batchnorm/efficient_net_w_hid_best_mixup_1M.pkl',
    f'../saved_models/dnns_mixup_1M_batchnorm/inception_w_hid_best_mixup_1M.pkl',
    f'../saved_models/dnns_mixup_1M_batchnorm/res_net_w_hid_best_mixup_1M.pkl',
     ]

model_with_hidden = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)

model_with_hidden.load(path_to_densenet=dnn_paths[0])
model_with_hidden.load(path_to_efficientnet=dnn_paths[1])
model_with_hidden.load(path_to_inception=dnn_paths[2])
model_with_hidden.load(path_to_resnet=dnn_paths[3])

preds = []
truth = []


for m in [model_with_hidden.densenet, model_with_hidden.efficientnet, model_with_hidden.inception, model_with_hidden.resnet]:
    
    preds = []
    truth = []

    for i in range(len(test_data)):
    
        img_p = test_data.loc[i, 'image_path']
        #img = plt.imread(img_p)

        pred = m.predict(img_p, device='cpu')#ff_model.predict(img_p)
        #print(pred)

        #print(pred[1][1], '-', test_data.loc[i, 'target'])
        preds.append(pred[1][1])
        truth.append(test_data.loc[i, 'target'])

    preds = (np.array(preds) > 0.5).astype(int)
    truth = np.array(truth)

    print('=========================')
    print(confusion_matrix(truth, preds))
    print('=========================')


'''
par_dir = '../../../data/isic_2020' 
data_path = os.path.join(par_dir, 'ISIC_2020_Training_GroundTruth.csv')
data_df = pd.read_csv(data_path)

pos_imgs = data_df.loc[data_df['target'] == 1]
neg_imgs = data_df.loc[data_df['target'] == 0].sample(n=len(pos_imgs))
print(len(pos_imgs))
print(len(neg_imgs))

print("Inferring New Images...")
for i, row in neg_imgs.iterrows():

    img_id = row['image_name']
    img_path = os.path.join(par_dir, f'train/{img_id}.jpg')
    show_image(img_path)
    print(img_path)
    pred, probs = model_with_hidden.predict(img_path)
    print(pred)
    print() 
'''
