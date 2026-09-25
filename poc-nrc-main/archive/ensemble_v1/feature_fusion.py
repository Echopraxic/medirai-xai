from re import A
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from trainer import Trainer
from torch import nn, optim

from isic_data_loader import ISICDataset, ISICMixupDataset, ISICFeatureFusionMixupDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision.transforms import v2

class MediraiFeatureFusionEnsemble(nn.Module):

    def __init__(self, 
            dense_net=None,
            efficient_net=None, 
            inception=None, 
            res_net=None, 
            h_feat_size=256,
            h_fusion_size=512, 
            nb_classes=2,
            device='cuda'
        ):

        super(MediraiFeatureFusionEnsemble, self).__init__()

        self.h_feat_size = h_feat_size
        self.h_fusion_size = h_fusion_size

        self.dense_net = dense_net
        self.efficient_net = efficient_net
        self.inception = inception
        self.res_net = res_net

        for param in self.dense_net.parameters():
            param.requires_grad = False
        for param in self.efficient_net.parameters():
            param.requires_grad = False
        for param in self.inception.parameters():
            param.requires_grad = False
        for param in self.res_net.parameters():
            param.requires_grad= False

        #self._lock_all_dnn_grads()
        self._gen_all_feature_extractors()
        self._gen_classifier(nb_classes)

        self.device = device


    def dense_net_feat_forward(self, x):
        
        return self.dense_net_extractor(x.clone())['classifier.1']
    
    def efficient_net_feat_forward(self, x):
 
        return self.efficient_net_extractor(x.clone())['classifier.1.1']

    def inception_feat_forward(self, x):
        
        return self.inception_extractor(x.clone())['fc.1']

    def res_net_feat_forward(self, x):
        
        return self.res_net_extractor(x.clone())['fc.1']

    def forward(self, x_224, x_299):
        
        x_dense = self.dense_net_feat_forward(x_224)
        x_efficient = self.efficient_net_feat_forward(x_224)
        x_inception = self.inception_feat_forward(x_299)
        x_res_net = self.res_net_feat_forward(x_224)

        x = torch.cat((x_dense, x_efficient, x_inception, x_res_net), dim=1)
        x = self.classifier(x)
        
        return x

    def _gen_all_feature_extractors(self):

        self.dense_net_extractor = create_feature_extractor(
            self.dense_net,
            return_nodes=['classifier.1']
        )

        self.efficient_net_extractor = create_feature_extractor(
            self.efficient_net,
            return_nodes = ['classifier.1.1']
        )

        self.inception_extractor = create_feature_extractor(
            self.inception,
            return_nodes = ['fc.1']
        )

        self.res_net_extractor = create_feature_extractor(
            self.res_net,
            return_nodes = ['fc.1']
        )    

    def _lock_all_dnn_grads(self):

        self.dense_net.lock_grads()
        self.efficient_net.lock_grads()
        self.inception.lock_grads()
        self.res_net.lock_grads()

    def _gen_classifier(self, nb_classes):

        self.classifier = nn.Sequential(
            nn.Linear(self.h_feat_size*4, self.h_fusion_size),
            nn.ReLU6(),
            nn.Dropout(0.5),
            nn.Linear(self.h_fusion_size, nb_classes)
        )


class FFModel:

    def __init__(self, model):

        self.model = model

    def load(self, path_to_state_dict, device):

        self.model.load_state_dict(
                            torch.load(
                                path_to_state_dict, 
                                map_location=torch.device(device),
                                ),
                            )
        self.model.eval()

    
    def process_image(self, img, img_size):

        test_transforms = v2.Compose([
                v2.Resize(img_size),
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
        img_to_pred_244 = self.process_image(input, (224, 224))
        img_to_pred_299 = self.process_image(input, (299, 299))
        img_to_pred_244 = img_to_pred_244.to(device)
        img_to_pred_299 = img_to_pred_299.to(device)
        scores = self.model(img_to_pred_244, img_to_pred_299).detach().cpu().numpy()[0]

        return scores, softmax(scores)
    


    
class FeatureFusionTrainer():

    def __init__(self,
                 training_csv,
                 model,
                 model_name='ff_model_XYYX.pkl',
                 device='cuda',
                 ):
        
        self.train_df = pd.read_csv(training_csv)
        self.model = model
        self.model_name = model_name
        self.device = device
    
    def data_transforms(self, image_size):

        transform={
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),    
            transforms.RandomApply([
                transforms.RandomRotation(degrees=30)
                ], p=0.5),
            transforms.ColorJitter(
                    brightness=0.2,  
                    contrast=0.1,    
                    saturation=0.5,  
                    hue=0.1          
                ),
            #transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val':transforms.Compose([
            #transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        'post':transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        return transform   

    def train(self,
              num_epochs=20,
              learning_rate = 1e-4,
              weight_decay = 3e-5,
              batch_size=32):
        
        mask = np.random.rand(len(self.train_df)) < 0.8
        train = self.train_df[mask]
        val = self.train_df[~mask]

        transform = self.data_transforms((299,299))
        trainDataset = ISICFeatureFusionMixupDataset(
                            train, 
                            (224,224), 
                            img_size_2=(299,299), 
                            transform=transform['train'],
                            as_tensor=transform['post'])
        valDataset = ISICFeatureFusionMixupDataset(
                            val, 
                            (224,224), 
                            img_size_2=(299,299), 
                            transform=None,
                            as_tensor=transform['post'])

        train_loader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )

        val_loader = DataLoader(
            valDataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=2
        )
    
        optimizer = optim.AdamW(
                            self.model.parameters(), 
                            lr=learning_rate,
                            weight_decay=weight_decay,
                        )
        criterion = nn.CrossEntropyLoss()
        
        iter = 0
        for epoch in range(num_epochs):

            print(f'epoch {epoch+1}/{num_epochs}')

            correct = 0
            total = 0
            for i, (images_244, images_299, labels) in enumerate(tqdm(train_loader)):
                
                if self.device == 'cuda':
                    images_244, images_299, labels = images_244.to('cuda'), images_299.to('cuda'), labels.to('cuda')
                
                optimizer.zero_grad()
                outputs = self.model(images_244, images_299)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()
                iter = iter + 1

            accuracy = 100 * correct / total
            print(f'Train accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}')
            #if epoch % 10 == 0:
                        
            correct = 0
            total = 0
            # Iterate through test dataset
            for images_244, images_299, labels in tqdm(val_loader):

                outputs = self.model(images_244, images_299)
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()

            accuracy = 100 * correct / total
            print(f'Val accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}')
            if self.model_name is None:
                print('model_name is None' )
                torch.save(self.model.state_dict(), f'./saved_models/fusion/feature_fusion_epoch-{epoch}.plk')
            else:
                val_acc = accuracy.detach().numpy()
                print('Saving', self.model_name.replace('XYYX', str(epoch)))
                torch.save(self.model.state_dict(), self.model_name.replace('XYYX', str(epoch)).replace('.plk', f'-val_acc_{val_acc:.4f}.pkl')) 


from medirai_base_model import MediraiEnsembleModelV1

if __name__ == '__main__':
    
    dnn_paths = [
        f'../../saved_models/2025-06-12/dense_net_w_hid_best_mixup_new_negs.pkl',
        f'../../saved_models/2025-06-12/efficient_net_w_hid_best_mixup_new_negs.pkl',
        f'../../saved_models/2025-06-12/inception_w_hid_1_mixup_new_negs.pkl',
        f'../../saved_models/2025-06-12/res_net_w_hid_best_mixup_new_negs.pkl',
    ]
    
    medirai_model = MediraiEnsembleModelV1(device='cpu', use_hidden_layers=True)
    medirai_model.load(path_to_densenet=dnn_paths[0])
    medirai_model.load(path_to_efficientnet=dnn_paths[1])
    medirai_model.load(path_to_inception=dnn_paths[2])
    medirai_model.load(path_to_resnet=dnn_paths[3])

    medirai_ff_model = MediraiFeatureFusionEnsemble(
        medirai_model.densenet.model,
        medirai_model.efficientnet.model,
        medirai_model.inception.model,
        medirai_model.resnet.model,
        256,
        512,
        device='cpu'
    )

    ff_helper = FFModel(medirai_ff_model)
    ff_helper.load('../../saved_models/ff_large_network/OLD_new_ff_model_19.pkl', device='cpu')
    print(ff_helper.predict('./sample_img.jpg', 'cpu')) 
    #train_csv = './train_data_EQ.csv'
    #train_csv = './mixup_train_data.csv'
    #medirai_ff_trainer = FeatureFusionTrainer(train_csv, medirai_ff_model, device='cpu')
    #medirai_ff_trainer.train()

    #INFERENCE LOOP
    from sklearn.metrics import confusion_matrix
    test_data = pd.read_csv('./test_EQ.csv')
    
    preds = []
    truth = []
    
    for i in range(len(test_data)):

        img_p = test_data.loc[i, 'image_path']
        #img = plt.imread(img_p)

        pred = ff_helper.predict(img_p, device='cpu')#ff_model.predict(img_p)
        #print(pred)

        #print(pred[1][1], '-', test_data.loc[i, 'target'])
        preds.append(pred[1][1])
        truth.append(test_data.loc[i, 'target'])

    preds = (np.array(preds) > 0.5).astype(int)
    truth = np.array(truth)

    print('=========================')
    print(confusion_matrix(truth, preds))
    print('=========================')


