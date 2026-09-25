import json

from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import pandas as pd
import os

class MediraiBiomedClip:

    def __init__(self, model_path):

        self.model_path = model_path
        self.loaded_model, self.tokenizer, self.preprocess = self._load_model()


    def _load_model(self):
        
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
  

    def predict(self, image):

        if isinstance(image, str):
            image = Image.open(image)

        img = torch.stack([self.preprocess(image)])
        text = 

