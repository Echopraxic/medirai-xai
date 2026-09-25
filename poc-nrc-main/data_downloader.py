
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import cv2
import pickle



class DataDownloader():

    def __init__(self,
                 src_url,
                 download_obj,
                 verbose=True
                 ):
        
        self.src_url = src_url
        self.download_obj = download_obj
        self.verbose = verbose
        self.target_dir = None
        self.files = None
        

    def _download(self):

        if os.path.exists(self.download_obj):

            print(f'Download object {self.download_obj} already exists on disc. Will not proceed with download.')

        else:

            if self.verbose:
                print(f'Attempting to download from {self.src_url}...')
            
            urllib.request.urlretrieve(self.src_url, self.download_obj)

            if self.verbose:
                print(f'\t...success!')

    
    def _unzip(self, target_dir):

        if os.path.exists(target_dir):
            
            print(f'Target directory {target_dir} already exits. Will not proceeed with unarchiving.')

        else:

            with zipfile.ZipFile(self.download_obj) as zip_ref:
                zip_ref.extractall(target_dir)


    def _set_target_dir(self, target_dir):

        self.target_dir = target_dir


    def _set_files(self, files):

        self.files = files

    
    def _walk_data_dirs(self):

        return [w for w in  os.walk(self.target_dir, topdown=True)]


    def download_unzip_walk(self):

        target_dir = os.path.splitext(self.download_obj)[0]
        
        if self.target_dir is None:
            self._set_target_dir(target_dir)

        self._download()
        self._unzip(self.target_dir)
        
        files = self._walk_data_dirs()

        if self.files is None:
            self._set_files(files)


    def _resize_image(self, img):

        if img.shape[0] < self.resize_images[0]:
            img = cv2.resize(img, self.resize_images, interpolation=cv2.INTER_CUBIC) #for enlarging
        elif img.shape[0] > self.resize_images[0]: 
            img = cv2.resize(img, self.resize_images, interpolation=cv2.INTER_AREA) #for shrinking
        return img

    
class IsicDataLoader(DataDownloader):

    def __init__(self,
                 src_url,
                 download_obj,
                 truths_url,
                 load_from_file = None, #'./pickled_isic_data.pkl',
                 save_data = './pickled_isic_data.pkl', #None to not save
                 resize_images = (128,128) #None will not resize 
                ):

        super().__init__(src_url, download_obj)
        self.download_unzip_walk()
        self.truths_url = truths_url
        self.base_path = os.path.join(self.files[0][0], self.files[0][1][0])
        self.save_data = save_data
        self.resize_images = resize_images
        self.meta = self._load_meta()

        if load_from_file is None:

            self.labels = self._load_labels()
            self.images = self._load_images()
        
        else:

            images, labels = self._load_images_label_from_disc()
            self.labels = labels
            self.images = images

        if (not save_data is None) and (load_from_file is None):

            self._save_images_labels()


    def download_labels(self):
        
        out_name = self.truths_url.split('/')[-1]
        out_path = os.path.join(self.base_path, out_name)

        if not os.path.exists(out_path):
            try:
                urllib.request.urlretrieve(self.truths_url, out_path)
            except Exception as e:
                raise print(f"Unable to access ISIC permissive ground truths.")

        return out_path


    def _get_img_paths_via_meta(self):

        img_ids = self.meta['isic_id']
        img_paths = [os.path.join(self.base_path, im_id+'.jpg') for im_id in img_ids]
        return img_paths
    

    def _load_labels(self):

        return np.array(self.meta['malignant'])
        

    def _load_images(self):

        img_paths = self._get_img_paths_via_meta()
        images = [plt.imread(p) for p in img_paths]

        if self.resize_images is not None:
            images = np.array([self._resize_image(img) for img in images])

        return images


    def _load_meta(self):

        meta_fname = [f for f in self.files[1][-1] if f.endswith('csv') and 'meta' in f][0]
        meta_path = os.path.join(self.base_path, meta_fname)
        meta_df = pd.read_csv(meta_path)

        labels_path = self.download_labels()
        labels_df = pd.read_csv(labels_path)

        all_meta_df = meta_df.merge(labels_df, how='left', on='isic_id')
        return all_meta_df
    
    
    def _load_images_label_from_disc(self):

        with open(self.save_data, 'rb') as f:
            d = pickle.load(f)

        return d['images'], d['labels']
         

    def _save_images_labels(self):

        with open(self.save_data, 'wb') as f:
            pickle.dump(
                {'images':self.images, 'labels':self.labels},
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )


    def __getitem__(self, idx):
        return self.images[idx], self.label[idx]


    def __len__(self):
        return len(self.images)


class MelonomaCancerDataLoader(DataDownloader):

    def __init__(self,
                 src_url,
                 download_obj,
                 save_as_npy = None,
                 shuffle = True,
                 resize_images = None
                ):
         
        super().__init__(src_url, download_obj)
        self.download_unzip_walk()
        self.shuffle = shuffle
        self.resize_images = resize_images
        self.base_path = os.path.join(self.files[0][0], self.files[0][1][0])
        self.meta = self._make_meta()
        self.train_images, self.train_labels = self._load_set(is_train=1)
        self.test_images, self.test_labels = self._load_set(is_train=0)

        print(self.train_images.shape)


    def _dir_data_to_dict(self, par, files, is_malig, is_train):

        return [ {'path':os.path.join(par, f), 'fname':f, 'malignant': is_malig, 'is_train':is_train} for f in files ]


    def _make_meta(self):

        meta_path = os.path.join(self.base_path,'metadata.csv') 
        if os.path.exists(meta_path):
            return pd.read_csv(meta_path)
    
        tests_malig_meta = self._dir_data_to_dict(self.files[3][0], self.files[3][2], 1, 0)
        tests_not_malig_meta = self._dir_data_to_dict(self.files[4][0], self.files[4][2], 0, 0)
        train_malig_meta = self._dir_data_to_dict(self.files[6][0], self.files[6][2], 1, 1)
        train_not_malig_meta = self._dir_data_to_dict(self.files[7][0], self.files[7][2], 0, 1)

        meta = tests_malig_meta+tests_not_malig_meta+train_malig_meta+train_not_malig_meta
        meta_df = pd.DataFrame(meta)

        if self.shuffle:
            meta_df = meta_df.sample(frac=1.0)

        meta_df.to_csv(meta_path, index=False)

        return meta_df 
    

    def _load_set(self, is_train=1):

        img_paths = self.meta['path'].loc[self.meta['is_train'] == is_train]
        labels = self.meta['malignant'].loc[self.meta['is_train'] == is_train] 

        imgs = np.array([plt.imread(ip) for ip in img_paths])

        if self.resize_images != None:
            imgs = np.array([self._resize_image(im) for im in imgs])

        lbls = np.array(labels)

        return imgs, lbls


class ImageAugmentor():

    def __init__():

        pass

    def digital_hair_removal(img):

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
        kernel = cv2.getStructuringElement(1, (13,13))
        black_hat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
        ret, thresh = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)
        
        return dst


if __name__ == '__main__':

    isic_url_permissive = 'https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Permissive_Training_Input.zip'
    isic_url_truths = 'https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Permissive_Training_GroundTruth.csv'

    melonoma_cancer_url = 'https://www.kaggle.com/api/v1/datasets/download/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images'

    isic_permissive_out_obj = './data/isic_archive.zip'
    melonoma_cancer_out_obj = './data/melonoma_cancer.zip'

    isic_data = IsicDataLoader(isic_url_permissive, isic_permissive_out_obj, isic_url_truths)
    melo_data = MelonomaCancerDataLoader(melonoma_cancer_url, melonoma_cancer_out_obj)

    
    


