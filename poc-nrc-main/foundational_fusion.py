from biomed_clip import MediraiBiomedClip
from derm_foundation import MediraiDermFoundation

from predictor_network import Predictor, PredictorNet, PredictorNetTrainer

import pandas as pd
import numpy as np

import torch

class FoundationalFusion:

    def __init__(
            self,
            clip_prednet_path: str,
            derm_prednet_path: str,
            fusion_input_size: int,
            fusion_hidden_size: int,
            device='cuda',
            clip_prednet_input_size=512,
            clip_prednet_hidden_size=128,
            derm_prednet_input_size=6144,
            derm_prednet_hidden_size=128,
            ):
        
        self.device = device
        self.clip_pred_net = PredictorNet(
                clip_prednet_input_size,
                clip_prednet_hidden_size,
                )
        self.clip_pred_model = MediraiBiomedClip(
                    None,
                    self.device,
                    self.clip_pred_net,
                    pred_path = clip_prednet_path
                )
        
        self.derm_pred_net = PredictorNet(
                derm_prednet_input_size,
                derm_prednet_hidden_size, 
                )
        self.derm_pred_model = MediraiDermFoundation(
                    None,
                    self.device,
                    self.derm_pred_net,
                    pred_model_path=derm_prednet_path
                )
        self.fusion_prednet = Predictor(fusion_input_size, fusion_hidden_size)
    
    
    def get_embedding(self, img):

        _, clip_emb = self.clip_pred_model.get_img_features(img)
        derm_emb = self.derm_pred_model.gen_embedding_from_path(img)

        clip_hid = self.clip_pred_net.get_hidden(
                clip_emb
                ).detach().numpy()
        derm_hid = self.derm_pred_net.get_hidden(
                torch.from_numpy(derm_emb)
                ).detach().numpy()

        fused = np.concat((clip_hid, derm_hid))

        return fused

    
    def get_n_embeddings(self, train_df: pd.DataFrame, embedding_path, label_path):
        
        print(f'generating {len(train_df)} features...')
        fusion_feats = []
        labels = []
        for i in range(len(train_df)):

            img_path = train_df.loc[i, 'image_path']
            fused = self.get_embedding(img_path)
            fusion_feats.append(fused)
            labels.append(int(train_df.loc[i, 'target']))

        np.save(embedding_path, np.array(fusion_feats)) 
        np.save(label_path, np.array(labels))
        print('\t...done!')
        

    def load(self, path_to_frozen):
        
        self.fusion_prednet.load(path_to_frozen, self.device)
        
    def train(self, x, y, savename='foundfusion', **kwargs):
        
        self.fusion_prednet.train(x, y, savename, **kwargs)
    
    def predict(self, img):
        
        emb = self.get_embedding(img)
        emb = torch.from_numpy(emb)
        return self.fusion_prednet.predict(emb)
        
    def get_hidden(self, img):
        
        emb = self.get_embedding(img)

        return self.fusion_prednet.get_hidden(emb)

if __name__ == '__main__':

    found_fusion_model = FoundationalFusion(
                '../prod_models/found_biomed_clip_predictor_net_epoch-10.plk',
                '../prod_models/found_derm_foundation_predictor_net_epoch-10.plk',
                256,
                128,
                device='cpu'
            )
    
    train_csv = './train_test_csvs/train_data_EQ_new_negs.csv'
    train_df = pd.read_csv(train_csv)
    img_paths = train_df['image_path']
    corrected_paths = [p.replace('../../../', '../../') for p in img_paths]
    train_df['image_path'] = corrected_paths

    print(train_df.head())
    
    embs_path = '../embeddings/found_fusion_embs.npy'
    lbls_path = '../embeddings/found_fusion_lbls.npy'
    
    found_fusion_model.get_n_embeddings(train_df, embs_path, lbls_path)
    
    x = np.load(embs_path)
    y = np.load(lbls_path)

    found_fusion_model.train(train_df, x, y, 'found_fusion')

