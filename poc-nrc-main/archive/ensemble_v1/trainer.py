
import copy
import torch
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from isic_data_loader import ISICDataset, ISICMixupDataset
import numpy as np
import datetime

from torch import nn

import pandas as pd

class Trainer:

    def __init__(self,
                 model_name, 
                 model,
                 config, 
                 criterion, 
                 optimizer,
                 train_df, 
                 scheduler=None, 
                 device='cuda',
                 #is_inception=False
                 ):
        
        self.train_datetime = datetime.datetime.now()
        self.model_name = model_name
        self.model = model.to(device)
        self.CONFIG = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_df = train_df
        self.device = device
        self.best_model = {'wts':None, 'acc':0.0}
        self.history={'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        

    def accuracy(self, y_true, y_pred):

        acc = (y_pred==y_true).float().mean()
        return acc


    def train_step(self, train_loader, gradient_accumulation):

        epoch_loss = 0.0
        epoch_acc = 0.0
        
        
        self.model.train()
        for i, (x, y) in enumerate(tqdm(train_loader, total=len(train_loader))):
            
            if self.device=='cuda':
                x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            if self.model_name == 'inception':
                output, aux_outputs = self.model(x)
                loss_1 = self.criterion(output, y.long())
                loss_2 = self.criterion(aux_outputs, y.long())
                loss = loss_1 + 0.4*loss_2
            
            else:
                output = self.model(x)
                loss = self.criterion(output, y.long())

            if gradient_accumulation > 1.0:
                loss = loss / gradient_accumulation
                
            loss.backward()

            if (i+1)%gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                

            y_pred = output.argmax(dim=1)
            acc = self.accuracy(y, y_pred)

            epoch_loss += loss.item()*gradient_accumulation
            epoch_acc += acc

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        
        return epoch_loss, epoch_acc


    def eval_step(self, val_loader):

        epoch_loss = 0.0
        epoch_acc = 0.0
        
        self.model.eval()
        for x, y in tqdm(val_loader, total=len(val_loader)):
            if self.device=='cuda':
                x, y = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                output = self.model(x)
                loss = self.criterion(output, y.long())

            y_pred = output.argmax(dim=1)
            acc = self.accuracy(y, y_pred)

            epoch_loss += loss
            epoch_acc += acc

        epoch_loss /= len(val_loader)
        epoch_acc /= len(val_loader)
        
        return epoch_loss, epoch_acc


    def data_transforms(self):

        transform={
        'train': transforms.Compose([
            transforms.Resize((self.CONFIG['image_size'], self.CONFIG['image_size'])),
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
            # EdgeEnhancementTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val':transforms.Compose([
            transforms.Resize((self.CONFIG['image_size'], self.CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        return transform


    def train_csv_to_dataloader(self, train_csv):

        return ISICDataset(train_df, self.data_transforms)


    def update_best_model(self, epoch_acc):

        if epoch_acc > self.best_model['acc']:
            self.best_model['acc'] = epoch_acc
            self.best_model['wts'] = copy.deepcopy(self.model.state_dict())


    def save_best_model(self):

        out_path = self.CONFIG['freeze_name_template'].replace('XYYX','best')
        torch.save(self.best_model['wts'], out_path)


    def save_model_state_dict(self, out_path):

        torch.save(self.model.state_dict(), out_path)


    def save_training_log(self):

        train_log_df = pd.DataFrame(self.history)
        train_log_df.to_csv(f'./logs/train_{self.model_name}_at_{self.train_datetime.strftime("%Y%m%d-%H%M%S")}.csv', index=False)


    def fit(self, 
            eval_per_epoch: int=1, 
            gradient_accumulation: int=1):
        
        skf = StratifiedKFold(
                n_splits=self.CONFIG['n_fold'], 
                shuffle=True, 
                random_state=self.CONFIG['seed'])

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.train_df, self.train_df['target'])):
            print(f"================Fold {fold + 1}/{self.CONFIG['n_fold']}===============")

            train = self.train_df.iloc[train_idx].reset_index(drop=True)
            val = self.train_df.iloc[val_idx].reset_index(drop=True)

            class_weight = compute_class_weight(
                                'balanced',
                                classes=np.unique(train['target']),
                                y=train['target']
                            )
            class_weight = torch.FloatTensor(class_weight).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)

            transform = self.data_transforms()
            trainDataset = ISICMixupDataset(train, (self.CONFIG['image_size'], self.CONFIG['image_size']),  transform['train'])
            valDataset = ISICMixupDataset(val, (self.CONFIG['image_size'], self.CONFIG['image_size']), transform['val'])

            train_loader = DataLoader(
                trainDataset,
                batch_size=self.CONFIG['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=2
            )

            val_loader = DataLoader(
                valDataset,
                batch_size=self.CONFIG['batch_size'],
                drop_last=False,
                num_workers=2
            )

            for epoch in range(self.CONFIG['num_epochs']):

                loss, acc = self.train_step(train_loader, gradient_accumulation=gradient_accumulation)
                self.history['train_loss'].append(loss)
                self.history['train_acc'].append(acc.cpu().item())
    
                print(f'EPOCH: {epoch+1}/{self.CONFIG['num_epochs']}: train loss: {loss:.4f}, train accuracy: {acc:.4f}')

                if epoch%eval_per_epoch==0 and eval_per_epoch <= self.CONFIG['num_epochs']:

                    loss, acc = self.eval_step(val_loader)
                    self.history['val_loss'].append(loss.cpu().item())
                    self.history['val_acc'].append(acc.cpu().item())
                    self.update_best_model(acc)
                    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}')
    
                if self.scheduler:
                    self.scheduler.step(loss)

            self.save_model_state_dict(self.CONFIG['freeze_name_template'].replace('XYYX',f'{str(fold+1)}'))

        self.save_best_model()
        self.save_training_log()

