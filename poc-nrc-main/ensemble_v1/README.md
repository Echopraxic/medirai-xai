# Ensemble Model
This package will fine-tune DenseNet, EfficientNet, Inception, and ResNet to the task of malignant skin lesion. Fine-tuned models have been provided, but have been trained on limited data.

Dependencies are located in `requirements.txt` and this code was developed in `Python 3.12.6`.

Trained EfficientNet model had to be broken up into to pieces to be pushed to this repo. To reconstruct it, in bash call:
```bash
cat efficientnet_compressed.z* > efficeintnet.zip
unzip efficeintnet.zip
```

# Authors
 - Tyson Mitchell (tyson.mitchell@nrc-cnrc.gc.ca)
 - Erfan Fatehi (erfan.fatehi@nrc-cnrc.gc.ca)

# Data
The data used to train this model can be downloaded by calling `python data_downloader.py`. Note that this script expects the presence of a `data` directory in the same location as where the script is called. 

To recreate the provided trained models, the downloaded file `ISIC_2024_Permissive_Training_GroundTruth.csv` can be combined with `./utils/hand_filtered_malig_egs.csv` to create the training meta dataset. See Section Training for details on further requires for the training `csv`.


# Usage
The core module `medirai_base_model.py` has three core functionalities:
  - Sequentially fine-tune the four modules constructing the ensemble,
  - Inference or predicting if a new image is malignant, and
  - Used GradCAM provide some insight into model explanability.
    
In the current state of the code, all directories present in this respository are required. Further work is required to ensure file directory structure is as expected.

## Training 
To retrain (or re-fine-tune) models a `csv` of specific format must be supplied. The csv must have columns `image_id`', `target` (malignant or not), and `image_path` (the path to training image). See `train_data_TESTING.csv` for an example. 

To train the mdoels:
```python
model = MediraiEnsembleModelV1(device='cuda')
train_csv = './train_data_TESTING.csv'
model.train(train_csv, to_train='all')
```

## Inference 
After training, models are not loaded by default. For inference, first models must be loaded:
```python
model = MediraiEnsembleModelV1(device='cuda')
print('Loading Pretained Models...')
model.load(path_to_densenet='./trained_models/densenet_fold10_state_dict.pkl')
model.load(path_to_efficientnet='./trained_models/efficientnet_fold10_state_dict.pkl')
model.load(path_to_inception='./trained_models/inception_fold10_state_dict.pkl')
model.load(path_to_resnet='./trained_models/resnet_fold10_state_dict.pkl')
print(model)
```
Once the models are loaded, inference is conducted as
```python
sample_img = './sample_img.jpg'

probs, preds = model.predict(sample_img)
print(preds)
>>> [0 0 1 0]
print(probs)
>>> [4.0121503e-02 1.0206775e-02 5.2957815e-01 6.3474258e-06]
```
Inference results are returned in alphabetic order, or `[DenseNet, EfficientNet, Inception, ResNet]`. 

## GradCAM
GradCAM can provide some insights into the explanable of a model. First, load the models (see previous sectoin), then to predict with GradCAM, use:
```Python
sample_img, sample_is_malig = './sample_img.jpg', 0
pred_cam_img = model.predict_with_cam(
                 sample_img,
                 sample_is_malig,
                 save_path='./grad_cam_results/test_image.png'
               )
```
If `save_path` an image will be written to disc, for example:

![GradCAM Example](grad_cam_results/test_image.png)

# Recommendations:
Each model has a default configuration that is stored as an attribute:
```python
model = MediraiEnsembleModelV1(device='cuda')
model.load(path_to_densenet='./trained_models/densenet_fold10_state_dict.pkl')
model.densenet.CONFIG
```
The default configurations are verbose, and not all parameters are required (as of now) but may be required later on. These configuration can be found in `medirai_model.py`. It is receommended to set `n_fold` to `2` and `num_epochs` to `2` for all models to test functionality on a new system. 

# List of Know Issues
- Lack of type hints
- Lack of DocStrings
- Lack of file structure input/output validation and automation

# Generating Synthetic Data
Derm-T2IM can be leveraged to generate synthetic data. Please reference: https://arxiv.org/html/2401.05159v1/#bib.bib9 for detail.

The authors of this model have made it available under MIT license. First install `diffusers` and `transformers` packages from HuggingFace. Then, to use the model:
```python
from diffusers import DiffusionPipeline
url = 'https://huggingface.co/MAli-Farooq/Derm-T2IM/blob/main/Derm-T2IM.safetensors'
pipeline = StableDiffusionPipeline.from_single_file(url)
prompt = "Sample of benign moles"
image = pipeline(prompt).images[0]
```
Note that this is resource intensive and hardware dependend can take 5-10mins to generate a single image. 
