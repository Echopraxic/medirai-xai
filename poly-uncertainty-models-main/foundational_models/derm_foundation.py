import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image 
from io import BytesIO
import numpy as np


import matplotlib.pyplot as plt

class DermFoundation:

    def __init__(self, model_path):

        self.model_path = model_path
        self.loaded_model = self._load_model()
        #print(self.loaded_model)


    def _gen_embedding(self, img_bytes):

        input_tensor = self._format_input(img_bytes)
        embedding = self._call_model(input_tensor)
        return embedding


    def _gen_img_bytes(self, pil_img):

        buf = BytesIO()
        pil_img.convert('RGB').save(buf, 'PNG')
        img_bytes = buf.getvalue()

        return img_bytes


    def gen_embedding_from_path(self, img_path):

        img = Image.open(img_path)
        img_bytes = self._gen_img_bytes(img)
        embedding = self._gen_embedding(img_bytes)

        return embedding


    def gen_embedding_from_array(self, img):

        img = Image.fromarray(img)
        img_bytes = self._gen_img_bytes(img)
        embedding = self._gen_embedding(img_bytes)

        return embedding
    

    def _format_input(self, image_bytes):
       
       return tf.train.Example(features=tf.train.Features(
            feature={'image/encoded': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes]))
            })).SerializeToString()


    def _load_model(self):

        return tf.keras.layers.TFSMLayer(self.model_path, call_endpoint='serving_default')


    def _call_model(self, input_tensor): 

        output = self.loaded_model(tf.constant([input_tensor]))
        return output['embedding'].numpy().flatten()

import joblib
import umap
class DermFoundUmap:

    def __init__(self, umap_save_path):

        self.path = umap_save_path
        self.umap_model = self._load_saved_model()
        print(type(self.umap_model))

    def _load_saved_model(self):

        return joblib.load(self.path)


    def reduce(self, emb):

        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
            return self.umap_model.transform(emb)
        else:
            return self.umap_model.transform(emb)
    
    def classify(self, emb):

        umap_emb = self.reduce(emb)
        umap_emb = umap_emb[:,0]
        preds = (umap_emb < 6).astype(int)
        
        return preds


if __name__ == "__main__":
    
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