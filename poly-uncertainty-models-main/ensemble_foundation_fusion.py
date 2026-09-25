

from os import wait
import numpy as np
import pandas as pd

import torch

from feature_fusion import FeatureFusionTrainer, MediraiFeatureFusionEnsemble, FFModel
from predictor_network import PredictorNet, PredictorNetTrainer, Predictor
from medirai_base_model import MediraiEnsembleModelV1

from derm_foundation import MediraiDermFoundation
from biomed_clip import MediraiBiomedClip

from torchvision.transforms import v2

class EnsembleFoundationFusion:

    def __init__(
            self,
            path_to_ensemble_fusion,
            path_to_foundational_fusion,
            paths_to_dnns,
            global_fusion_input,
            global_fusion_hidden,
            global_fusion_dropout=0.9,
            ensemble_input_size=256,
            ensemble_hidden_size = 128,
            foundational_input_size=256,
            foundational_hidden_size=128,
            device='cuda',
            path_to_frozen = None
            ):
        
        self.device = device

        print(f'Loading ensemble fusion model...')
        self.ensemble_fusion_model = self._load_ensemble_fusion_model(
                                                path_to_ensemble_fusion,
                                                paths_to_dnns,
                                                ensemble_input_size,
                                                ensemble_hidden_size)
        
        print(f'Loading foundational fusion model...')
        self.foundational_fusion_model = self._load_foundational_fusion_model(
                                            path_to_foundational_fusion,
                                            foundational_input_size,
                                            foundational_hidden_size
                                        )
        self.fusion_model = self._init_fusion_model(
                                    global_fusion_input, 
                                    global_fusion_hidden,
                                    global_fusion_dropout)
        if path_to_frozen is not None:
            self.fusion_model.load(path_to_frozen, self.device)

        clip_pred_net = PredictorNet(512, 128)
        self.biomed_clip = MediraiBiomedClip(None, 'cpu', clip_pred_net, pred_path = './foundational_models/biomed_clip_predictor_net_epoch-19.plk')

        found_pred_net = PredictorNet(6144, 128)
        self.derm_foundation = MediraiDermFoundation(None, 'cpu', found_pred_net, pred_model_path='./foundational_models/derm_foundation_predictor_net_epoch-19.plk')
        

    def _load_ensemble_fusion_model(self, path_to_ensemble_fusion, dnn_paths, input_size, hidden_size):
        
        medirai_model = MediraiEnsembleModelV1(device=self.device, use_hidden_layers=True)
        medirai_model.load(path_to_densenet=dnn_paths[0])
        medirai_model.load(path_to_efficientnet=dnn_paths[1])
        medirai_model.load(path_to_inception=dnn_paths[2])
        medirai_model.load(path_to_resnet=dnn_paths[3])
        ensemble_model = MediraiFeatureFusionEnsemble(  
                    medirai_model.densenet.model,
                    medirai_model.efficientnet.model,
                    medirai_model.inception.model,
                    medirai_model.resnet.model,
                    input_size,
                    hidden_size
                )
        ensemble_fusion_model = FFModel(ensemble_model)
        ensemble_fusion_model.load(path_to_ensemble_fusion, device=self.device)
        return ensemble_fusion_model

    def _load_foundational_fusion_model(self, path_to_ensemble_fusion, input_size, hidden_size):

        foundational_fusion = Predictor(input_size, hidden_size)
        foundational_fusion.load(path_to_ensemble_fusion, self.device)
        return foundational_fusion

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

    def get_foundational_hiddens(self, img_path):

        derm_found_emb = self.derm_foundation.gen_embedding_from_path(img_path)
        derm_found_hidden = self.derm_foundation.pred_model.get_hidden(
                torch.from_numpy(derm_found_emb)
                ).detach().numpy()

        _, biomed_clip_emb = self.biomed_clip.get_img_features(img_path)
        biomed_clip_hidden = self.biomed_clip.pred_model.get_hidden(
                biomed_clip_emb
                ).detach().numpy()

        return derm_found_hidden, biomed_clip_hidden

    
    def get_foundational_fusion_hidden_from_embs(self, derm_hid, clip_hid):

        foundational_fusion_input = np.concat((clip_hid, derm_hid))
        ffh = self.foundational_fusion_model.get_hidden(
                torch.from_numpy(foundational_fusion_input)
                ).detach().numpy()
        return ffh 


    def get_foundational_fusion_hiddens(self, img_path):

        derm_hid, clip_hid = self.get_foundational_hiddens(img_path)
        foundational_fusion_hid = self.get_foundational_fusion_hidden_from_embs(derm_hid, clip_hid)

        return foundational_fusion_hid


    def fuse_hidden(self, x):
        
        ens_hid = self.ensemble_fusion_model.get_hidden(x, self.device)
        fon_hid = self.get_foundational_fusion_hiddens(x)

        return np.concatenate((ens_hid, fon_hid))

    def gen_embeddings(self, train_df, embeddings_outpath: str, labels_outpath: str):

        embeddings = []
        labels = []

        for i in range(len(train_df)):
            
            img_path = train_df.loc[i, 'image_path']
            embeddings.append(self.fuse_hidden(img_path))
            labels.append(train_df.loc[i, 'target'])

        np.save(embeddings_outpath, np.array(embeddings))
        np.save(labels_outpath, np.array(labels))

    def _init_fusion_model(self, input_size, hidden_size, dropout):

        return Predictor(input_size, hidden_size, dropout_prob=dropout)

    def train(self, x, y, save_name):

        if isinstance(x, str):
           x = np.load(x)
        if isinstance(y, str):
            y = np.load(y)

        if not isinstance(x, np.ndarray):
            TypeError(f'First argument is not instance of np.ndarray!')
        if not isinstance(y, np.ndarray):
            TypeError(f'Second argument is not instance of np.ndarray!')

        self.fusion_model.train(x, y, save_name=save_name)


    def predict(self, x):

        hidden_emb = self.fuse_hidden(x)
        return self.fusion_model.predict(torch.from_numpy(hidden_emb)).detach().numpy()
        


if __name__ == '__main__':
    
    dnn_paths = [
        f'../prod_models/dnn_dense_net_w_hid_best_mixup_1M.pkl',
        f'../prod_models/dnn_efficient_net_w_hid_best_mixup_1M.pkl',
        f'../prod_models/dnn_inception_w_hid_best_mixup_1M.pkl',
        f'../prod_models/dnn_res_net_w_hid_best_mixup_1M.pkl',
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
        128,
        device='cpu'
    )

    path_to_ensemble_fusion = f'../prod_models/fus_dnn_fusion_epoch-10.pkl'
    path_to_foundational_fusion = f'../prod_models/fus_foundation_fusion_epoch-10.plk'
    
    global_fusion = EnsembleFoundationFusion(
                path_to_ensemble_fusion,
                path_to_foundational_fusion,
                dnn_paths,
                256,
                64,
                device='cpu'
            )
    
    train_df = pd.read_csv('./train_test_csvs/train_data_EQ.csv')
    img_paths = train_df['image_path']
    corrected_paths = [p.replace('../../../', '../../') for p in img_paths]
    train_df['image_path'] = corrected_paths

    emb_path = './global_fusion_embs.npy'
    lbl_path = './global_fusion_lbls.npy'
    save_name = 'global_fusion'
    
    print('Generating fusion embeddings...')
    global_fusion.gen_embeddings(train_df, emb_path, lbl_path)
    global_fusion.train(emb_path, lbl_path, save_name)
    
    print(f'Performing Tests...')
    from sklearn.metrics import confusion_matrix

    test_df = pd.read_csv('./ensemble_v1/test_EQ.csv')
    img_paths = test_df['image_path']
    corrected_paths = [p.replace('../../../', '../../') for p in img_paths]
    test_df['image_path'] = corrected_paths

    fusion_preds = []
    clip_preds = []
    derm_preds = []
    truth = []

    for i in range(len(test_df)):
        
        img_p = test_df.loc[i, 'image_path']
        p = global_fusion.predict(test_df.loc[i, 'image_path']) 
        fusion_preds.append(np.argmax(p)) 
        truth.append(int(test_df.loc[i,'target']))


    print('Fusion Results')
    print(confusion_matrix(truth, fusion_preds))

