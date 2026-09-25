

from os import wait
import numpy as np
import numpy.typing as npt
import pandas as pd

import torch

from PIL import Image

from dnn_feature_fusion import FeatureFusionTrainer, MediraiFeatureFusionEnsemble, FFModel
from predictor_network import PredictorNet, PredictorNetTrainer, Predictor
from medirai_base_model import MediraiEnsembleModelV1

from derm_foundation import MediraiDermFoundation
from biomed_clip import MediraiBiomedClip

from torchvision.transforms import v2

class EnsembleFoundationFusion:
    '''
    Global fusion model

    Attributes
    ----------
    path_to_ensemble_fusion : str
        path to frozen DNN feature fusion model
    path_to_foundational_fusion : str
        path to frozen foundational fusion model
    paths_to_dnns : list[str]
        paths to frozen DNNs, in alphabetic order
    path_to_dermfound: str
        path to frozen DermFound model
    path_to_biomedclip : str
        path to frozen BiomedClip model
    global_fusion_input : int 
        input size of MLP post feature fusion proceedure
    global_fusion_hidden : int
        size of hidden layer for MLP  
    global_fusion_dropout : float
        dropout rate of MLP
    ensemble_input_size : int
        size of input for DNN feature fustion MLP network
    ensemble_hidden_size : int 
        size of hidden layer of DNN feature fustion MLP network
    foundational_input_size : int
        size of input for foundational fusion MLP network
    foundational_hidden_size : int 
        size of hidden layer of foundational fusion MLP network
    device : str
        device to map global fusion model too. Must be 'cpu' or 'cuda'
    path_to_frozen = None
    '''
    def __init__(
            self,
            path_to_ensemble_fusion,
            path_to_foundational_fusion,
            paths_to_dnns,
            path_to_dermfound,
            path_to_biomedclip,
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
        self.biomed_clip = MediraiBiomedClip(None, 'cpu', clip_pred_net, pred_path=path_to_biomedclip)

        found_pred_net = PredictorNet(6144, 128)
        self.derm_foundation = MediraiDermFoundation(None, 'cpu', found_pred_net, pred_model_path=path_to_dermfound)
        

    def _load_ensemble_fusion_model(self, 
                                    path_to_ensemble_fusion : str, 
                                    dnn_paths : list[str], 
                                    input_size : int, 
                                    hidden_size: int) -> FFModel: 

        '''
        Helper function to load DNN fusion model from frozen

        Parameters
        ----------
        path_to_ensemble_fusion : str
            path to frozen DNN feature fusion model
        dnn_paths : list[str]
            paths to frozen DNNs, in alphabetic order
        input_size : int
            size of input for DNN feature fustion MLP network
        hidden_size : int 
            size of hidden layer of DNN feature fustion MLP network
        '''
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

    def _load_foundational_fusion_model(self, 
                                        path_to_foundational_fusion : str, 
                                        input_size : int, 
                                        hidden_size : int) -> Predictor:
        '''
        Helper function to load foundational fusion network
        from frozen

        Parameters
        ----------
        path_to_foundational_fusion : str
            path to frozen foundational fusion model
        input_size : int
            size of input for foundational fusion MLP network
        hidden_size : int 
            size of hidden layer of foundational fusion MLP network
        '''
        foundational_fusion = Predictor(input_size, hidden_size)
        foundational_fusion.load(path_to_foundational_fusion, self.device)
        return foundational_fusion

    def process_image(self, img : str | npt.NDArray, img_size : tuple[int, int]) -> torch.Tensor:
        '''
        Preprocess image for network pass through

        Parameters
        ----------
        img : str | npt.NDArray
            path to image or image as array
        image_size : tuple[int, int]
            new height and width image will be adjusted too

        Returns
        -------
        img_to_pred : torch.Tensor
            image ready to fed into pytorch models
        '''
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

    def get_foundational_hiddens(self, img_path : str) -> tuple[npt.NDArray, npt.NDArray]:
        '''
        Get hidden layers for DermFoundation and BiomedClip

        Parameters
        ----------
        img_path : str
            image to retrieve hiddens for

        Returns
        -------
        hidden : tuple[npt.NDArray, npt.NDArray]
            values of hidden layers in MLPs for foundational model
            MLPs
        '''
        derm_found_emb = self.derm_foundation.gen_embedding_from_path(img_path)
        derm_found_hidden = self.derm_foundation.pred_model.get_hidden(
                torch.from_numpy(derm_found_emb)
                ).detach().numpy()

        _, biomed_clip_emb = self.biomed_clip.get_img_features(img_path)
        biomed_clip_hidden = self.biomed_clip.pred_model.get_hidden(
                biomed_clip_emb
                ).detach().numpy()

        return derm_found_hidden, biomed_clip_hidden

    
    def get_foundational_fusion_hidden_from_embs(self, derm_hid : npt.NDArray, clip_hid : npt.NDArray) -> npt.NDArray:
        '''
        Get hidden layers from Foundational Fusion MLP

        Parameters
        ----------
        derm_hid : npt.NDArray
            hidden values from DermFoundation MLP
        clip_hid : npt.NDArray
            hidden values from BiomedClip MLP

        Returns
        -------
        ffh : tuple[npt.NDArray, npt.NDArray]
            values of hidden for foundational fusion model
            MLP
        '''
        foundational_fusion_input = np.concat((clip_hid, derm_hid))
        ffh = self.foundational_fusion_model.get_hidden(
                torch.from_numpy(foundational_fusion_input)
                ).detach().numpy()
        return ffh 


    def get_foundational_fusion_hiddens(self, img_path : str) -> npt.NDArray:
        '''
        Get hidden layer from foundational fusion provided
        and image path

        Parameters
        ----------
        image_path : str
            path to image
        
        Returns
        -------
        foundational_fusion_hid : npt.NDArray
            hidden layer for foundational fusion MLP
        '''

        derm_hid, clip_hid = self.get_foundational_hiddens(img_path)
        foundational_fusion_hid = self.get_foundational_fusion_hidden_from_embs(derm_hid, clip_hid)

        return foundational_fusion_hid


    def fuse_hidden(self, x : str) -> npt.NDArray:
        '''
        Fuse the hidden layers of foundational fusion
        MLP and DNN Esemble MLP

        Parameters
        ----------
        x : str
            path to input image

        Returns
        -------
        fusion : npt.NDArray
            fused hidden layers
        '''
        ens_hid = self.ensemble_fusion_model.get_hidden(x, self.device)
        fon_hid = self.get_foundational_fusion_hiddens(x)

        return np.concatenate((ens_hid, fon_hid))

    def gen_embeddings(self, train_df : pd.DataFrame, embeddings_outpath: str, labels_outpath: str) -> None:
        '''
        Generate many embeddings for on of DermFoundation and BiomedClip

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data frame. Must have "image_path" and
            "target" as columns.
        embeddings_outpth : str
            Write location for embeddings
        labels_outpath : str
            Wrate location for labels. Embeddings and labels pairs are
            accessed by index.
        '''
        embeddings = []
        labels = []

        for i in range(len(train_df)):
            
            img_path = train_df.loc[i, 'image_path']
            embeddings.append(self.fuse_hidden(img_path))
            labels.append(train_df.loc[i, 'target'])

        np.save(embeddings_outpath, np.array(embeddings))
        np.save(labels_outpath, np.array(labels))

    def _init_fusion_model(self, input_size : int, hidden_size : int, dropout : float) -> Predictor:
        '''
        Helper fuction to initialize final MLP
        
        Paramters
        ---------
        input_size : int
            input size for the MLP
        hidden_size : int
            size of hidden layer of the MLP
        dropout : float
            dropout rate of the MLP

        Return
        ------
        MLP : PredictorNet
            MLP network
        '''

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

        self.fusion_model.train(x, y, save_name)


    def predict(self, x:str) -> npt.NDArray:
        '''
        Peform an inference call on the Global Fusion network

        Parameters
        ----------
        x : str
            image to perform inference call on.

        Returns
        -------
        logits : npt.NDArray
            output logits of global fusion network
        '''

        hidden_emb = self.fuse_hidden(x)
        return self.fusion_model.predict(torch.from_numpy(hidden_emb)).detach().numpy()
        


if __name__ == '__main__':
    
    # Example Usage
    dnn_paths = [
        f'../saved_models/dnns_mixup_1M_batchnorm/dense_net_w_hid_best_mixup_1M.pkl',
        f'../saved_models/dnns_mixup_1M_batchnorm/efficient_net_w_hid_best_mixup_1M.pkl',
        f'../saved_models/dnns_mixup_1M_batchnorm/inception_w_hid_best_mixup_1M.pkl',
        f'../saved_models/dnns_mixup_1M_batchnorm/res_net_w_hid_best_mixup_1M.pkl',
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
        1024,
        128,
        device='cpu'
    )

    path_to_ensemble_fusion = f'../saved_models/dnns_mixup_1M_batchnorm/new_ff_model_19.pkl'
    path_to_foundational_fusion = f'../prod_models/fus_foundation_fusion_epoch-10.plk'
    
    global_fusion = EnsembleFoundationFusion(
                path_to_ensemble_fusion,
                path_to_foundational_fusion,
                dnn_paths,
                '../prod_models/found_derm_foundation_predictor_net_epoch-10.plk',
                '../prod_models/found_biomed_clip_predictor_net_epoch-10.plk',
                256,
                64,
                device='cpu',
                path_to_frozen='global_fusion_epoch-13.plk'
            )
    ''' path_to_ensemble_fusion,
            path_to_foundational_fusion,
            paths_to_dnns,
            path_to_dermfound,
            path_to_biomedclip,
            global_fusion_input,
            global_fusion_hidden,
            
    '''

    train_df = pd.read_csv('./train_test_csvs/train_data_EQ.csv')[:10]
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

