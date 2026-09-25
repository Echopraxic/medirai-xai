import json

import numpy as np
import numpy.typing as npt
import typing 

from tqdm import tqdm

from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import pandas as pd
import os

from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from predictor_network import PredictorNet, PredictorNetTrainer

class MediraiBiomedClip:
    '''
    Wrapper class for Microsoft's BiomedCLIP transformer model. For more information
    and instuctions on accessing the model visit:

    https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

    Attributes
    ----------
    clip_path : None
        Deprecated attribute
    device : device
        Device to run the transformer model on 'cpu' or 'cuda'
    pred_model : PredictorNet
        MLP that resides on top of the embeddings produced by the CLIP model
        to yeild binary classifications.
    pred_path : str[Optional]
        path of trained MLP to loaded trained models from

    '''

    def __init__(self, clip_path : None, 
                 device : str, 
                 pred_model : PredictorNet, 
                 pred_path : typing.Optional[str]=None):

        self.clip_path = clip_path
        self.device = device
        self.clip_model, self.clip_tokenizer, self.clip_preprocess = self._load_clip_model()
        self.default_prompt = 'this is a photo of a skin lesion'

        self.pred_model = pred_model
        if pred_path is not None:
             self.load_pred_net(pred_path)

    def _load_clip_model(self) -> tuple:
        '''
        Load the clip model from source. This may take some time when running
        for the first time.

        Returns
        -------
        model
            Actual clip model
        tokenizer
            Clip prompt tokenizer, not used in out case 
        clip_preprocess
            data propress to prepare data for inference calls with the
            clip model.
        
        '''

        hf_hub_download(
            repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            filename="open_clip_pytorch_model.bin",
            local_dir="checkpoints"
        )
        hf_hub_download(
            repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            filename="open_clip_config.json",
            local_dir="checkpoints"
        )
        
        model_name = "biomedclip_local"

        with open("checkpoints/open_clip_config.json", "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        model, _, preprocess = create_model_and_transforms(
                    model_name = model_name,
                    pretrained="checkpoints/open_clip_pytorch_model.bin",
                    **{f"image_{k}": v for k, v in preprocess_cfg.items()}
                )

        tokenizer = get_tokenizer(model_name)

        return model, tokenizer, preprocess
  

    def get_img_features(self, image_path : str, prompt : typing.Optional[str]=None) -> tuple:
        '''
        Get embedding representaitons fora single image

        Parameters
        ----------
        image : str
            path to image to get the embedding for
        prompt : str
            Clip model requries a prompt to get a final prediction. We are 
            mainly using the embeddings.

        Returns
        -------
        preds
            predictions from the clip model
        image_features
            image embedding
        '''

        if isinstance(image_path, str):
            image = Image.open(image_path)
        
        if prompt is None:
            prompt = self.default_prompt

        img = torch.stack([self.clip_preprocess(image)])
        text = self.clip_tokenizer(prompt, context_length=256).to(self.device)

        with torch.no_grad():
            
            image_features, text_features, logits_scale = self.clip_model(img, text)
            logits = (logits_scale*image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)

            logits = logits.cpu().numpy()[0]
            sorted_indices = sorted_indices.cpu().numpy()[0]

            preds = np.argmax(sorted_indices)
        
        return preds, image_features[0]

    def gen_n_image_features(self,
                             training_csv : str, 
                             image_data_storage : str = './biomedclip_embs.npy',
                             label_data_storage : str ='./biomedclip_lbls.npy',
                             ) -> None:
        '''
        Batch run retrieving embeddings for a collection of images in a 
        training csv format. Csv must have 'target' and 'image_path'
        columns

        Parameters
        ----------
        training_csv : str
            data containing information of generating the embeddings. Must be a 'standard' 
            training csv with columns target and image_path
        image_data_storage : str
            path to location where image embeddings will be saved. Embeddings will be saved
            as a numpy array
        label_data_storage : str
            path to location where labels will be saved. Image-label pairs are accessed by
            coinciding indicies. 
        '''
        
        train_df = pd.read_csv(training_csv)
        embs = []
        lbls = []

        print(f'Generating feature representations...')
        for i in tqdm(range(len(train_df))):
            
            label = int(train_df.loc[i, 'target'])
            img_p = train_df.loc[i, 'image_path']
            
            preds, img_feats = self.get_img_features(img_p)
            embs.append(img_feats)
            lbls.append(label)

        embeddings = np.array(embs)
        labels = np.array(lbls)

        np.save(image_data_storage, embeddings)
        np.save(label_data_storage, labels)


    def train_pred_net(self,  feat_reps_path : str, labels_path : str ) -> None:
        '''
        Using generated image embeddings train a predictor network to predict
        and embedding as one of benign or malignant.

        Parameters
        ----------
        feat_reps_path : str
            path to the generated embeddings for the training data set. Should
            be a frozen numpy array.  
        label_path : str
            path to the generated labels that accompany the trainint data. 
            Shoud be a frozen numpy array.
        '''
        
        feat_reps = np.load(feat_reps_path)
        labels = np.load(labels_path)
        trainer = PredictorNetTrainer(
            self.pred_model,
            feat_reps,
            labels,
            'biomed_clip_predictor_net',
        ) 
        trainer.train()
    

    def load_pred_net(self, path : str) -> None:
        '''
        Load pretrained predictor network that sits on top of BiomedClip

        Parameters
        ----------
        path : str
            Path to frozen predictor network weights as state dictionary. 
        '''
        
        self.pred_model.load_state_dict(torch.load(path, weights_only=True))
        self.pred_model.eval()  # otherwise Dropout(0.5) stays active at inference (CODEBASE_TODO P0-15)
        

    def predict(self, img : str) -> npt.NDArray:
        '''
        Do inference through BiomedClip and trained predictor network.

        Parameters
        ----------
        img : str
            Path to image to perform inference one.

        Returns
        -------
        preds : npt.NDArray
            predictions for each class
        '''

        _, img_feats = self.get_img_features(img)
        with torch.no_grad():
            preds = self.pred_model(img_feats)
        #probs = preds.softmax()

        return preds 


if __name__ == '__main__':

    # Example Usage:
    pred_net = PredictorNet(512, 128)
    mbmc = MediraiBiomedClip(
                None,
                'cpu',
                pred_net,
            )
    
    train_csv = './train_data_EQ.csv'
    feat_p = './embs/bmc_embs.npy'
    lbls_p = './embs/bmc_lbls.npy'
    
    mbmc.gen_n_image_features(train_csv,
                    image_data_storage= feat_p,
                    label_data_storage= lbls_p)

    mbmc.train_pred_net(feat_p, lbls_p)



