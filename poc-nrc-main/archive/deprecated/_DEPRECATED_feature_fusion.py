
import pandas as pd
import numpy as np
import os
import copy

import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import TensorDataset, DataLoader

from medirai_base_model import MediraiEnsembleModelV1
from sklearn.model_selection import train_test_split

from PIL import Image
from tqdm import tqdm

class PredictNetwork(nn.Module):

    def __init__(self, h_dnn, h_fusion):
        
        x_size = int(h_dnn*4)
        super(PredictNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(x_size, h_fusion),
            nn.ReLU6(),
            nn.Dropout(0.5),
            nn.Linear(h_fusion, 2)
        )

    def forward(self, feat_rep):
        #print(feat_rep.shape)
        return self.classifier(feat_rep)


class FeatureFusion:

    def __init__(self,
                 dnn_hid = 256,
                 prednet_hid = 512,
                 temp_dir = './ff_temp',
                 #train_device = 'cuda',
                 #inference_device='cuda',
                 training_csv=None):
        
        self.model_loaded = False
        self.dnn_names = ['dense_net', 'efficient_net', 'inception', 'res_net'] 
        self.dnn_hid = dnn_hid
        self.prednet_hid = prednet_hid
        self.temp_dir = temp_dir
        self.FF_CONFIGS = self.get_ff_configs()
        self.medirai_ensemble = MediraiEnsembleModelV1(configs=self.FF_CONFIGS)
        self.pred_net = PredictNetwork(dnn_hid, prednet_hid)
        self.training_csv = training_csv


    def get_ff_configs(self):

        configs = []
        img_sizes = [224, 224, 299, 224]
        for i, m_name in enumerate(self.dnn_names):

            CONFIG = {
                'seed': 42,
                'image_size': img_sizes[i],
                'num_classes': 2,
                'batch_size':32,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'n_fold': 2,
                'learning_rate': 1e-5,
                'gamma': 0.95,
                'momentum': 0.9,
                'num_epochs': 1,
                'freeze_name_template': os.path.join(self.temp_dir, f'{m_name}-h_{self.dnn_hid}-XYYX.pkl'),
                'grad_acc': 8,
                'weight_decay': 0.01,
                'hidden_size': self.dnn_hid
            }
            configs.append(CONFIG)
            
        return configs
    

    def predict(self, img):
        
        if not self.model_loaded:
            IOError(f'Feature Fusion model not loaded.')

        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        else:
            img = Image.fromarray(img)

        d_trans = self.data_transforms(self.FF_CONFIGS[0])
        e_trans = self.data_transforms(self.FF_CONFIGS[1])
        i_trans = self.data_transforms(self.FF_CONFIGS[2])
        r_trans = self.data_transforms(self.FF_CONFIGS[3])
       
        d_img = d_trans['val'](copy.deepcopy(img)).unsqueeze(0)
        e_img = e_trans['val'](copy.deepcopy(img)).unsqueeze(0)
        i_img = i_trans['val'](copy.deepcopy(img)).unsqueeze(0)
        r_img = r_trans['val'](copy.deepcopy(img)).unsqueeze(0)

        d_fr = self.medirai_ensemble.densenet.get_feature_representation(d_img, 'classifier.1')
        e_fr = self.medirai_ensemble.efficientnet.get_feature_representation(e_img, 'classifier.1.1')
        i_fr = self.medirai_ensemble.inception.get_feature_representation(i_img, 'fc.1')
        r_fr = self.medirai_ensemble.resnet.get_feature_representation(r_img, 'fc.1')

        feat_fusion = np.hstack( (d_fr, e_fr, i_fr, r_fr) )
        scores = self.pred_net(torch.from_numpy(feat_fusion).unsqueeze(0))
        scores = scores.detach().numpy()[0][0]
        
        def softmax(x): 
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) 
         
        return scores, softmax(scores)
        

    def load_dnn(self, dnn_paths):

        self.medirai_ensemble.load(path_to_densenet=dnn_paths[0])
        self.medirai_ensemble.load(path_to_efficientnet=dnn_paths[1])
        self.medirai_ensemble.load(path_to_inception=dnn_paths[2])
        self.medirai_ensemble.load(path_to_resnet=dnn_paths[3])
        self._prep_ensemble_models()
    
    def load_pred_net(self, pred_net_path):

        prep_net = PredictNetwork(self.dnn_hid, self.prednet_hid)
        prep_net.load_state_dict(torch.load(pred_net_path, weights_only=True))
        prep_net.eval()


    def load(self, dnn_paths, pred_net_path):

        self.load_dnn(dnn_paths)
        self.load_pred_net(pred_net_path)
        self.model_loaded = True
        

    def _prep_ensemble_models(self):

        self.medirai_ensemble.densenet.lock_grads()
        self.medirai_ensemble.efficientnet.lock_grads()
        self.medirai_ensemble.inception.lock_grads()
        self.medirai_ensemble.resnet.lock_grads()

        self.medirai_ensemble.densenet.set_feature_extractor('classifier.1')
        self.medirai_ensemble.efficientnet.set_feature_extractor('classifier.1.1')
        self.medirai_ensemble.inception.set_feature_extractor('fc.1')
        self.medirai_ensemble.resnet.set_feature_extractor('fc.1')


    def gen_feature_representations(self, 
                                    data_size_multiplier=5,
                                    feat_reps_out_name = 'feat_reps_og.npy',
                                    labels_out_name = 'labels_og.npy',
                                    ):

        self._prep_ensemble_models()
        feat_reps = []
        labels = []
        d_trans = self.data_transforms(self.FF_CONFIGS[0])
        e_trans = self.data_transforms(self.FF_CONFIGS[1])
        i_trans = self.data_transforms(self.FF_CONFIGS[2])
        r_trans = self.data_transforms(self.FF_CONFIGS[3])

        #print(self.medirai_ensemble.densenet.feature_extractor)
        train_df = pd.read_csv(self.training_csv)

        for ds in range(data_size_multiplier):
            print(f'Getting feature representations: {ds+1}/{data_size_multiplier}...')
            for i, row in tqdm(train_df.iterrows(), total=len(train_df)):

                img = row['image_path']
                #img_to_pred = self.data_transforms['train'](img_to_pred)
                #img_to_pred = img_to_pred.unsqueeze(0)
                lbl = row['target']

                img = Image.open(img).convert('RGB')
                d_img = d_trans['train'](copy.deepcopy(img)).unsqueeze(0)
                e_img = e_trans['train'](copy.deepcopy(img)).unsqueeze(0)
                i_img = i_trans['train'](copy.deepcopy(img)).unsqueeze(0)
                r_img = r_trans['train'](copy.deepcopy(img)).unsqueeze(0)

                d_fr = self.medirai_ensemble.densenet.get_feature_representation(d_img, 'classifier.1')
                e_fr = self.medirai_ensemble.efficientnet.get_feature_representation(e_img, 'classifier.1.1')
                i_fr = self.medirai_ensemble.inception.get_feature_representation(i_img, 'fc.1')
                r_fr = self.medirai_ensemble.resnet.get_feature_representation(r_img, 'fc.1')

                feat_fusion = np.hstack( (d_fr, e_fr, i_fr, r_fr) )

                feat_reps.append(feat_fusion)
                labels.append(lbl)

        feat_reps = np.array(feat_reps)
        labels = np.array(labels)
        np.save(os.path.join(self.temp_dir, feat_reps_out_name), feat_reps)
        np.save(os.path.join(self.temp_dir, labels_out_name), labels)


    def train_ensemble_networks(self):

        self.medirai_ensemble.train(self.training_csv, to_train='all')
        

    def run_training_pipeline(self):

        if self.training_csv is None:
            IOError('Training csv is None, no training csv was passed.')

        #self.
        #best_dnns = 

        self.gen_feature_representations()
        self.train_predictor_network()


    def data_transforms(self, CONFIG):

        transform={
        'train': transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
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
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        return transform
     

    def train_predictor_network(
                    self,
                    path_to_feat_reps, 
                    path_to_labels,
                    num_epochs=100,
                    val_split = 0.1,
                    lr = 1e-5,
                    weight_decay = 5e-6):

        train_data = np.load(path_to_feat_reps)
        train_data = np.array([np.hstack(s) for s in train_data])
        train_labels = np.load(path_to_labels)

        #print(train_data.shape)
        #print(train_labels.shape)
        #assert 0 == 1

        X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=val_split, random_state=42)

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        
        train_loader = DataLoader(train_dataset)
        val_loader = DataLoader(val_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
                            self.pred_net.parameters(), 
                            lr = lr,
                            weight_decay = weight_decay,
                        )
            
        iter = 0
        #num_epochs = 10
        print('Training Feature Fusion:')
        for epoch in range(num_epochs):
            print(f'epoch {epoch+1}/{num_epochs}')

            correct = 0
            total = 0
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                
                optimizer.zero_grad()
                outputs = self.pred_net(images)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()
                iter = iter + 1

            accuracy = 100 * correct / total
            print(f'Train accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}; loss: {loss:.4f}')
            
                        
            correct = 0
            total = 0
            for images, labels in tqdm(val_loader):

                outputs = self.pred_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()

            accuracy = 100 * correct / total
            print(f'Val accuracy epoch {epoch+1}: {accuracy.detach().numpy():.4f}')
            torch.save(self.pred_net.state_dict(), os.path.join(self.temp_dir, f'feature_fusion_epoch-{epoch}-og.plk'))

    
    
if __name__ == '__main__':

    training_csv = './train_data_EQ.csv'
    ff_model = FeatureFusion(
            #dnn_hid=32,
            #prednet_hid=64,
            training_csv=training_csv,
            temp_dir='../../saved_models/ff_temp')
    #ff_model.train_ensemble_networks()
    
    k = 4
    dnn_paths = [
            f'../../saved_models/dnn_models_mixup/run1/dense_net_w_hid_{k+1}.pkl',
            f'../../saved_models/dnn_models_mixup/run1/efficient_net_w_hid_{k+1}.pkl',
            f'../../saved_models/dnn_models_mixup/run1/inception_w_hid_{k+1}.pkl',
            f'../../saved_models/dnn_models_mixup/run1/res_net_w_hid_{k+1}.pkl',
                ]
    ff_nn_paths = [
            f'../../saved_models/dnn_new_negs/dense_net_w_hid_best_mixup_new_negs.pkl',
            f'../../saved_models/dnn_new_negs/efficient_net_w_hid_best_mixup_new_negs.pkl',
            f'../../saved_models/dnn_new_negs/inception_w_hid_best_mixup_new_negs.pkl',
            f'../../saved_models/dnn_new_negs/res_net_w_hid_best_mixup_new_negs.pkl',
        ]

    ff_nn_paths = [
            f'../../saved_models/dnns_sigmoid/dense_net_w_hid_32_best_mixup_new_negs_sigmoid.pkl',
            f'../../saved_models/dnns_sigmoid/efficient_net_w_hid_32_best_mixup_new_negs_sigmoid.pkl',
            f'../../saved_models/dnns_sigmoid/inception_w_hid_32_best_mixup_new_negs_sigmoid.pkl',
            f'../../saved_models/dnns_sigmoid/res_net_w_hid_32_best_mixup_new_negs_sigmoid.pkl',
        ]
    
    ff_nn_paths = [
            f'../../saved_models/medirai_models_with_hidden/dense_net_w_hid_256_best.pkl',
            f'../../saved_models/medirai_models_with_hidden/efficient_net_w_hid_256_best.pkl',
            f'../../saved_models/medirai_models_with_hidden/inception_w_hid_256_best.pkl',
            f'../../saved_models/medirai_models_with_hidden/res_net_w_hid_256_best.pkl',
        ]
    

    ff_model.load_dnn(ff_nn_paths)
    #ff_model.gen_feature_representations(data_size_multiplier=2) 

    ff_model.train_predictor_network(
        '../../saved_models/ff_mixup/feat_reps_og.npy',
        '../../saved_models/ff_mixup/labels_og.npy'
    )

    #ff_model.load_pred_net('../../saved_models/ff_temp/feature_fusion_epoch-9.plk')
    
    sample_img, sample_is_malig = './sample_img.jpg', 0

    scores, probs = ff_model.predict(sample_img)
    print(probs)

    




        




