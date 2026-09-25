from tqdm.gui import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image 
from io import BytesIO

import typing
import numpy as np
import numpy.typing as npt
import pandas as pd

import matplotlib.pyplot as plt
from predictor_network import PredictorNet, PredictorNetTrainer
from tqdm import tqdm

from huggingface_hub import from_pretrained_keras
import torch

class MediraiDermFoundation:
    '''
    Wrapper class for Google's DermFoundation transformer model. For more information
    and instuctions on accessing the model visit:

    https://huggingface.co/google/derm-foundation    

    Attributes
    ----------
    found_model_path : None
        Deprecated attribute
    device : str 
        Device to run the transformer model on 'cpu' or 'cuda'
    pred_model : PredictorNet
        MLP that resides on top of the embeddings produced by the CLIP model
        to yeild binary classifications.
    found_model : 
    pred_path : str[Optional]
        path of trained MLP to loaded trained models from
    ''' 

    def __init__(   self, 
                    found_model_path : None, 
                    device : str, 
                    pred_model : PredictorNet, 
                    pred_model_path : typing.Optional[str]=None):

        self.found_model_path = found_model_path
        self.device = device
        self.found_model = self._load_found_model()
        self.pred_model = pred_model
        
        self.pred_model_path = None
        if pred_model_path is not None:
            self._load_pred_net(pred_model_path)
        
        #print(self.loaded_model)


    def _gen_embedding(self, img_bytes : bytes) -> npt.NDArray:
        '''
        Provide a preprocess image, get DermFoundation Embedding

        Parameters
        ----------
        img_bytes : bytes 
            image as byte array to get embedding for

        Returns
        -------
        embedding : npt.NDArray
            corrisponding image embedding 
        '''

        input_tensor = self._format_input(img_bytes)
        embedding = self._call_model(input_tensor)
        return embedding


    def _gen_img_bytes(self, pil_img : Image.Image) -> bytes:
        '''
        Convert PIL image to bytes

        Parameters
        ----------
        pil_img : PIL.Image.Image
            image loades a PIL object

        Returns
        -------
        img_bytes: bytes
            image as bytes
        '''
        buf = BytesIO()
        pil_img.convert('RGB').save(buf, 'PNG')
        img_bytes = buf.getvalue()

        return img_bytes


    def gen_embedding_from_path(self, img_path : str) -> npt.NDArray:
        '''
        Helper function to perform processing steps to go from image path
        to DermFoundation Embdding

        Parameters
        ----------
        img_path : str
            Path to image

        Returns
        -------
        embedding : npt.NDArray
            embedding of image at path endpoint
        '''

        img = Image.open(img_path)
        img_bytes = self._gen_img_bytes(img)
        embedding = self._gen_embedding(img_bytes)

        return embedding


    def gen_embedding_from_array(self, img : npt.NDArray) -> npt.NDArray:
        '''
        Helper function to perform processing steps to from numpy
        array to DermFoundation Embedding
        
        Parameters
        ----------
        img : npt.NDArray
            image as array in the usual sense (H, W, C).

        Returns
        -------
        embedding : npt.NDArray
            DermFoundation embedding generated from the image in passed
            array
        '''

        img = Image.fromarray(img)
        img_bytes = self._gen_img_bytes(img)
        embedding = self._gen_embedding(img_bytes)

        return embedding
    

    def _format_input(self, image_bytes : bytes) -> npt.NDArray:
        '''
        Helper Function to format bytes to proper inference calls of 
        DermFoundation

        Parameters
        ----------
        image_bytes : bytes
            image as bytes
        Returns
        -------
        tensor : tf.tensor
            usable input for DermFoundation
        '''
       
        return tf.train.Example(features=tf.train.Features(
            feature={'image/encoded': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes]))
            })).SerializeToString()


    def _load_found_model(self):
        '''
        Helper function to load DermFoundation.

        Returns
        -------
        keras_model : type unkown
            pretrained DermFoundation model
        ''' 
        return from_pretrained_keras("google/derm-foundation")
        #return tf.keras.layers.TFSMLayer("./local_model", call_endpoint='serving_default')


    def _call_model(self, input_tensor : tf.Tensor) -> npt.NDArray: 
        '''
        Helper function to run inference and return only embeddings

        Parameters
        ----------
        input_tensor : tf.Tensor
            Preprocess image, ready for input into DermFoundation

        Returns
        -------
        embedding : npt.NDArray
            DermFoundation embeddings of input tensor
        '''

        output = self.found_model(tf.constant([input_tensor]))
        return output['embedding'].numpy().flatten()
    
    def _load_pred_net(self, path : str):
        '''
        Helper function to load pretrained predictorNetwork

        Parameters
        ----------
        path :  str
            File path to saved pytorch state dictionary
        '''
        self.pred_model.load_state_dict(torch.load(path, weights_only=True))
     
    def gen_n_image_features(self,
                             training_csv : str, 
                             image_data_storage : str ='./derm_found_embs.npy',
                             label_data_storage : str = './derm_found_lbls.npy',
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
            
            img_feats = self.gen_embedding_from_path(img_p)
            embs.append(img_feats)
            lbls.append(label)

        embeddings = np.array(embs)
        labels = np.array(lbls)

        np.save(image_data_storage, embeddings)
        np.save(label_data_storage, labels)


    def train_pred_net(self,  feat_reps_path : str, labels_path : str):
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
            'derm_foundation_predictor_net',
        ) 
        trainer.train()
    
    def predict(self, img):

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

        emb = self.gen_embedding_from_path(img)
        pred = self.pred_model(torch.from_numpy(emb)).detach().numpy()
        return pred

 
if __name__ == "__main__":
    
    #Example Usage
    pred_net = PredictorNet(6144, 128)
    mdf = MediraiDermFoundation(
                None,
                'cpu',
                pred_net,
                pred_model_path='./derm_foundation_predictor_net_epoch-19.plk'
            )
    
    train_csv = './train_data_EQ.csv'
    feat_p = './embs/df_embs.npy'
    lbls_p = './embs/df_lbls.npy'
    #mdf.gen_n_image_features(train_csv,
    #                image_data_storage = feat_p,
    #                label_data_storage = lbls_p)
    #mdf.train_pred_net(feat_p, lbls_p)

    from sklearn.metrics import confusion_matrix
    print(mdf.pred_model)

    test_data = pd.read_csv('./test_EQ.csv')
    for i in range(len(test_data)):

        img_p = test_data.loc[i, 'image_path']
        truth = test_data.loc[i, 'target']
        emb = mdf.gen_embedding_from_path(img_p)
        pred = mdf.pred_model(torch.from_numpy(emb)).detach().numpy()
        print(mdf.pred_model.get_hidden(torch.from_numpy(emb)))
        print(pred, truth)

    '''
    sample_img = './sample_img_1.jpg'
    derm_found = DermFoundation('./local_model')
    emb_1 = derm_found.gen_embedding_from_path(sample_img)
    print(emb_1)

    sample_img_array = np.array(plt.imread(sample_img))
    emb_2 = derm_found.gen_embedding_from_array(sample_img_array)
    print(emb_2)

    sample_img_1 = './sample_img_1.jpg'
    sample_img_2 = './sample_img_2.jpg'

    emb_3 = derm_found.gen_embedding_from_path(sample_img_2)
    print(emb_1)
    print(emb_3)

    umap_reducer = DermFoundUmap('./umap_dim_red.sav')
    print(umap_reducer.reduce(emb_1))
    print(umap_reducer.reduce(np.array([emb_1, emb_2, emb_3])))
    print(umap_reducer.classify(emb_2))
    print(umap_reducer.classify(np.array([emb_1, emb_2, emb_3])))
    '''
