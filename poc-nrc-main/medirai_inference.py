import typing
import numpy as np

from predictor_network import PredictorNet

from biomed_clip import MediraiBiomedClip
from derm_foundation import MediraiDermFoundation

from medirai_base_model import MediraiEnsembleModelV1

from dnn_feature_fusion import MediraiFeatureFusionEnsemble, FFModel
from global_fusion import EnsembleFoundationFusion
from foundational_fusion import FoundationalFusion


class MediraiInferenceEngine:

    '''
    Class to be used for making inference calls on all models at 
    all levels.

    Attributes
    ----------
    loaded_models : list
        list of models that have been loaded and can thus be used
        for inference calls
    device : str
        where to run the inference. Must be 'cpu' or 'cuda'
    clip : MediraiBiomedClip | None
        BiomedClip model ready for inference if not None
    derm : MedieraiDermFounation | None
        DermFoundation model ready for inference if not None
    dnns : MediraiEnsembleModelV1 | None
        DNN models ready for inference if not None
    dnn_fusion : FFModel | None
        DNN feature fusion model ready for inference if not None
    found_fusion : Foundational Fusion | None
        Foundational Fusion model ready for inference if not None
    global_fusion : EnsembleFoundationFusion | None
        Global Fusion model ready for inference if not None
    '''

    def __init__(self, device = 'cuda'):
       
        self.loaded_models : list = []
        self.device : str = device 
        self.clip : typing.Optional[MediraiBiomedClip] = None
        self.derm : typing.Optional[MediraiDermFoundation] = None
        self.dnns : typing.Optional[MediraiEnsembleModelV1] = None
        self.dnn_fusion : typing.Optional[FFModel] = None
        self.found_fusion : typing.Optional[FoundationalFusion] = None
        self.global_fusion : typing.Optional[EnsembleFoundationFusion] = None

    def load_clip(self, path : str, mlp_input_size : int = 512, mlp_hidden_size : int = 128) -> None:
        '''
        Load BiomedClip model from save

        Parameters
        ----------
        Path : str
            path to frozen state dictionary for BiomedClip
        mlp_input_size : int
            required parameter for predictor network
        mlp_hidden_size : int
            required parameter for predictor network
        '''
        clip_pred_net = PredictorNet(mlp_input_size, mlp_hidden_size)
        self.clip = MediraiBiomedClip(
                    None,
                    self.device,
                    clip_pred_net,
                    pred_path = path 
                )
        self.loaded_models.append('clip')

    
    def load_derm(self, path : str, mlp_input_size : int = 6144, mlp_hidden_size : int = 128) -> None:
        '''
        Load DermFoundation model from save

        Parameters
        ----------
        Path : str
            path to frozen state dictionary for DermFoundation 
        mlp_input_size : int
            required parameter for predictor network
        mlp_hidden_size : int
            required parameter for predictor network
        '''
        derm_pred_net = PredictorNet(mlp_input_size, mlp_hidden_size)
        self.derm = MediraiDermFoundation(
                    None,
                    self.device,
                    derm_pred_net,
                    pred_model_path = path 
                )
        self.loaded_models.append('derm')
    
    def load_dnns(self, dnn_paths : list[str], configs = None) -> None:
        '''
        Load DNN ensemble model from save

        Parameters
        ----------
        dnn_paths : list[str]
            paths to dnns by alphabetic order
        configs : None
            overwrite default config is necessary 
        '''

        if configs is not None:
            self.dnns = MediraiEnsembleModelV1(device = self.device, configs=configs)
        else:
            self.dnns = MediraiEnsembleModelV1(device = self.device)

        self.dnns.load(path_to_densenet=dnn_paths[0])
        self.dnns.load(path_to_efficientnet=dnn_paths[1])
        self.dnns.load(path_to_inception=dnn_paths[2])
        self.dnns.load(path_to_resnet=dnn_paths[3])
        self.loaded_models.append('dnns')
    
    def load_dnn_fusion(self, path : str, mlp_input_size : int = 256, mlp_hidden_size : int = 128) -> None:
        '''
        Load DNN fusion model from save

        Parameters
        ----------
        path : str
            path to frozen state dictionary for fusion model 
        mlp_input_size : int
            required parameter for predictor network
        mlp_hidden_size : int
            required parameter for predictor network
        '''

        if 'dnns' not in self.loaded_models:
            raise AttributeError('dnn models must be loaded before dnn fusion. dnn fuison must share the same base dnns as self.dnns')
        dnn_fusion_base = MediraiFeatureFusionEnsemble(  
                    self.dnns.densenet.model,
                    self.dnns.efficientnet.model,
                    self.dnns.inception.model,
                    self.dnns.resnet.model,
                    mlp_input_size,
                    mlp_hidden_size 
                )
        self.dnn_fusion = FFModel(dnn_fusion_base)
        self.dnn_fusion.load(path, self.device)
        self.loaded_models.append('dnn_fusion')

    def load_found_fusion(self, mlp_path : str, clip_path : str, derm_path : str, mlp_input_size : int = 256, mlp_hidden_size : int = 128) -> None:
        '''
        Load foundational fusion model from save

        Parameters
        ----------
        mlp_path : str
            path to from fusion MLP for foundational fusion model 
        clip_path : str
            path to frozen state dictionary for BiomedClip model 
        derm_path : str
            path to frozen state dictionary for DermFounation model 
        mlp_input_size : int
            required parameter for predictor network
        mlp_hidden_size : int
            required parameter for predictor network
        '''


        self.found_fusion = FoundationalFusion(
                clip_path,
                derm_path,
                mlp_input_size,
                mlp_hidden_size,
                device='cpu'
            )
        self.found_fusion.load(mlp_path)
        self.loaded_models.append('found_fusion')

    def load_global_fusion(
                    self, 
                    mlp_path : str, 
                    found_fusion_path : str, 
                    dnn_paths : list[str],
                    derm_path : str,
                    clip_path : str,
                    dnn_fusion_path : str, 
                    mlp_input_size : int = 256, 
                    mlp_hidden_size : int = 64 ) -> None:
        
        '''
        Load foundational fusion model from save

        Parameters
        ----------
        mlp_path : str
            path to fusion MLP 
        found_fusion_path : str
            path to foundational fusion model
        dnn_paths : list[str]
            paths to dnn model in alphabetic order
        clip_path : str
            path to frozen state dictionary for BiomedClip model 
        derm_path : str
            path to frozen state dictionary for DermFounation model 
        dnn_fusion_path : str
            path to frozen state dictionary for DNN fusion model
        mlp_input_size : int
            required parameter for predictor network
        mlp_hidden_size : int
            required parameter for predictor network
        '''

        self.global_fusion = EnsembleFoundationFusion(
                dnn_fusion_path,
                found_fusion_path,
                dnn_paths,
                derm_path,
                clip_path,
                mlp_input_size,
                mlp_hidden_size,
                device=self.device, path_to_frozen=mlp_path,)
        self.loaded_models.append('global_fusion')


    def predict(self, img : str) -> dict:
        '''
        Run inference on supplied image for all loaded models

        Parameters
        ----------
        img : str
            path to image to perform inference on
        
        Returns
        -------
        preds : dict
            predictions from all loaded models

        '''
        
        preds = {}

        if 'clip' in self.loaded_models:
            clip_pred = self.clip.predict(img).detach().numpy()
            preds['clip'] = np.argmax(clip_pred)

        if 'derm' in self.loaded_models:
            derm_pred = self.derm.predict(img)
            preds['derm'] = np.argmax(derm_pred)
            
        if 'dnns' in self.loaded_models:
            dnn_probs, dnn_preds = self.dnns.predict(img)
            preds['densenet'] = dnn_probs[0]
            preds['efficientnet'] = dnn_probs[1]
            preds['inception'] = dnn_probs[2]
            preds['resnet'] = dnn_probs[3]
        
        if 'dnn_fusion' in self.loaded_models:
            dnn_fusion_pred = self.dnn_fusion.predict(img, self.device)
            preds['dnn_fusion'] = np.argmax(dnn_fusion_pred)
        
        if 'found_fusion' in self.loaded_models:
            found_fusion_pred = self.found_fusion.predict(img).detach().numpy()
            preds['found_fusion'] = np.argmax(found_fusion_pred)

        if 'global_fusion' in self.loaded_models:
            global_fusion_pred = self.global_fusion.predict(img)
            preds['global_fusion'] = np.argmax(global_fusion_pred)

        return preds

if __name__ == '__main__':

    #Sample Usage    
    densenet_path = '../prod_models/dnn_dense_net_w_hid_best_mixup_1M.pkl'
    efficientnet_path = '../prod_models/dnn_efficient_net_w_hid_best_mixup_1M.pkl'
    inception_path = '../prod_models/dnn_inception_w_hid_best_mixup_1M.pkl'
    resnet_path = '../prod_models/dnn_res_net_w_hid_best_mixup_1M.pkl'
    dnn_paths = [densenet_path, efficientnet_path, inception_path, resnet_path]

    clip_path = '../prod_models/found_biomed_clip_predictor_net_epoch-10.plk'
    derm_path = '../prod_models/found_derm_foundation_predictor_net_epoch-10.plk'
    
    dnn_fusion_path = '../prod_models/fus_dnn_fusion_epoch-3.pkl'
    found_fusion_path = '../prod_models/fus_foundation_fusion_epoch-10.plk'
    global_fusion_path = '../prod_models/fus_global_fusion_epoch-10.plk'

    inf_eng = MediraiInferenceEngine(device='cpu')

    inf_eng.load_clip(clip_path)
    inf_eng.load_derm(derm_path)
    inf_eng.load_dnns(dnn_paths)

    inf_eng.load_dnn_fusion(dnn_fusion_path)
    inf_eng.load_found_fusion(found_fusion_path, clip_path, derm_path)
    inf_eng.load_global_fusion(
                global_fusion_path,
                found_fusion_path,
                dnn_paths,
                derm_path,
                clip_path,
                dnn_fusion_path
    )
    
    print(inf_eng.predict('./sample_data/sample_img.jpg'))

