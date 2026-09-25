from data_loader import IsicDataLoader, MelonomaCancerDataLoader
from models_iter1 import train_iter1, MelonomaModelIter1

if __name__ == '__main__':

    fetch_data = False

    if fetch_data:
        isic_url_permissive = 'https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Permissive_Training_Input.zip'
        isic_url_truths = 'https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Permissive_Training_GroundTruth.csv'

        melonoma_cancer_url = 'https://www.kaggle.com/api/v1/datasets/download/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images'

        isic_permissive_out_obj = './data/isic_archive.zip'
        melonoma_cancer_out_obj = './data/melonoma_cancer.zip'

        isic_data = IsicDataLoader(isic_url_permissive, isic_permissive_out_obj, isic_url_truths)
        melo_data = MelonomaCancerDataLoader(melonoma_cancer_url, melonoma_cancer_out_obj)

    md = train_iter1(
            './data/melonoma_cancer/melanoma_cancer_dataset/train',
            './data/melonoma_cancer/melanoma_cancer_dataset/test',
             MelonomaModelIter1(),
        )
    md.train_model()
    md.evaluate_model()