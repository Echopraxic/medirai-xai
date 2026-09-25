
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

from isic_data_loader import ISICDataset, ISICMixupDataset, ISICDatasetWMod, ISICDatasetSegmentation
import numpy as np
import datetime

from torch import nn

import pandas as pd

class Trainer:
    '''
    Training protocol for DNNs only.

    Attributes
    ----------
    train_datetime : datetime.datetime
        timestamp for when training began
    model_name : str
        name of the model being training
    model : torchvision model
        actual tochvision model to be fine-tuned
    CONFIG : dict
        model parameters to training
    Criterion : torch loss 
        will be overwritten nn.CrossEntropy loss
    Optimizer : torch optimizer
        torch otpimizer criteria
    Scheduler : torch schedular
        torch schedular to handle training oddities
    train_df : pd.DataFrame
        training data as DataFrame
    device : str
        where to do the training, 'cuda' recommended
    data_loader_name : str
        which data loader to use
    set_kwargs : dict | None
        parameters to segmationation dataloader, if using it
    best_model : dict
        dictionary to keep track of the current best model by accuracy
    history : dict
        training tracker

    Notes
    -----
    Type hints are omitted from this class do to the complicated
    nature of pytorch objects. The code is relatively self-explanatory,
    but will not be as user friendly in modern IDEs. This class is also
    not intended to be accessed directly by users.

    '''

    def __init__(self,
                 model_name, 
                 model,
                 config, 
                 criterion, 
                 optimizer,
                 train_df, 
                 scheduler=None, 
                 device='cuda',
                 data_loader='default',
                 seg_kwargs=None,
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
        self.data_loader_name = data_loader
        self.seg_kwargs = seg_kwargs 
        self.best_model = {'wts':None, 'acc':0.0}
        self.history={'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


    def get_data_loader(self, data, size, transform):
        '''
        Get the data loader by name
        '''
        print(self.data_loader_name)
        if self.data_loader_name == 'default':
            return ISICDataset(data, size, transform=transform)
        elif self.data_loader_name == 'artificially_inflated': 
            return ISICDatasetWMod(data, size, transform=transform)
        elif self.data_loader_name == 'segmentation':
            return ISICDatasetSegmentation(data, size, transform=transform, seg_kwargs=self.seg_kwargs)
        elif self.data_loader_name == 'mixup':
            return ISICMixupDataset(data, size, transform=transform)
        else:
            raise ValueError(f'Data Loader {self.data_loader_name} data loader passed.')

    def accuracy(self, y_true, y_pred):
        '''
        Compute accuracy score
        '''
        acc = (y_pred==y_true).float().mean()
        return acc


    def train_step(self, train_loader, gradient_accumulation):
        '''
        Complete on training step
        '''
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
        '''
        Perform on evaluation step
        '''

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
        '''
        Define the data training transforms
        '''
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

    def update_best_model(self, epoch_acc):
        '''
        Update the best model
        '''
        if epoch_acc > self.best_model['acc']:
            self.best_model['acc'] = epoch_acc
            self.best_model['wts'] = copy.deepcopy(self.model.state_dict())


    def save_best_model(self):
        '''
        Save the weights of the best model
        '''
        out_path = self.CONFIG['freeze_name_template'].replace('XYYX','best')
        torch.save(self.best_model['wts'], out_path)


    def save_model_state_dict(self, out_path):
        '''
        save a pytorch state dictionary
        '''
        torch.save(self.model.state_dict(), out_path)


    def save_training_log(self):
        '''
        write training log to disc
        '''
        train_log_df = pd.DataFrame(self.history)
        train_log_df.to_csv(f'./logs/train_{self.model_name}_at_{self.train_datetime.strftime("%Y%m%d-%H%M%S")}.csv', index=False)


    def fit(self,
            eval_per_epoch: int=1,
            gradient_accumulation: int=1):
        '''
        Do the full training loop.

        Folds are formed over *source images*, not CSV rows (CODEBASE_TODO P0-4/P0-5):
        mixup rows and 'artificially_inflated' rows reuse the same image many times, so a
        row-level split put the same image in train and val. Each fold also restarts from
        the initial weights/optimizer/scheduler state; previously one model kept training
        across folds, so fold k validated on images it had trained on in earlier folds.
        Validation always uses the real (un-mixed, un-modified) held-out images.
        '''
        init_model = copy.deepcopy(self.model.state_dict())
        init_optimizer = copy.deepcopy(self.optimizer.state_dict())
        init_scheduler = copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else None

        images = source_images(self.train_df)
        skf = StratifiedKFold(
                n_splits=self.CONFIG['n_fold'],
                shuffle=True,
                random_state=self.CONFIG['seed'])

        for fold, (_, val_idx) in enumerate(skf.split(images, images['target'])):
            print(f"================Fold {fold + 1}/{self.CONFIG['n_fold']}===============")

            self.model.load_state_dict(init_model)
            self.optimizer.load_state_dict(init_optimizer)
            if self.scheduler:
                self.scheduler.load_state_dict(init_scheduler)
            fold_best = {'wts': None, 'acc': 0.0}

            val = images.iloc[val_idx].reset_index(drop=True)
            train = rows_without_images(self.train_df, set(val['image_id']))

            class_weight = compute_class_weight(
                                'balanced',
                                classes=np.unique(train['target']),
                                y=train['target']
                            )
            class_weight = torch.FloatTensor(class_weight).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)

            transform = self.data_transforms()
            size = (self.CONFIG['image_size'], self.CONFIG['image_size'])
            trainDataset = self.get_data_loader(train, size, transform['train'])
            if self.data_loader_name == 'segmentation':
                valDataset = self.get_data_loader(val, size, transform['val'])
            else:
                valDataset = ISICDataset(val, size, transform=transform['val'])

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
                    if acc > fold_best['acc']:
                        fold_best = {'wts': copy.deepcopy(self.model.state_dict()), 'acc': acc}
                    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}')

                if self.scheduler:
                    self.scheduler.step(loss)

            # best epoch of this fold (was: last epoch)
            torch.save(fold_best['wts'], self.CONFIG['freeze_name_template'].replace('XYYX', f'{str(fold+1)}'))

        # 'best' = best fold's best epoch. Folds are now independent, but picking the max over folds
        # is still optimistic; report the per-fold validation accuracies, not this maximum.
        self.save_best_model()
        self.save_training_log()


def _image_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    '''(image_id column, image_path column) pairs used by a training CSV.'''
    if 'image_id_1' in df.columns:  # mixup CSVs
        return [('image_id_1', 'image_path_1'), ('image_id_2', 'image_path_2')]
    return [('image_id', 'image_path')]


def source_images(train_df: pd.DataFrame) -> pd.DataFrame:
    '''
    One row per distinct source image (image_id, image_path, target) in a training CSV,
    whether it is a plain, 'artificially_inflated' (repeated ids) or mixup (paired ids) CSV.
    '''
    parts = [train_df[[id_col, path_col, 'target']].set_axis(['image_id', 'image_path', 'target'], axis=1)
             for id_col, path_col in _image_columns(train_df)]
    images = pd.concat(parts).drop_duplicates('image_id').sort_values('image_id')
    return images.reset_index(drop=True)


def rows_without_images(train_df: pd.DataFrame, image_ids: set) -> pd.DataFrame:
    '''Training rows that use none of the given source images.'''
    keep = np.ones(len(train_df), dtype=bool)
    for id_col, _ in _image_columns(train_df):
        keep &= ~train_df[id_col].isin(image_ids).to_numpy()
    return train_df[keep].reset_index(drop=True)

