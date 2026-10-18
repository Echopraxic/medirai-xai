"""
Author: Zeinab Abboud, Aloys Portafaix
Dataloaders and dataset classes defined. 

dataset means and standard deviations computed as follows

  	from torch.utils.data import ConcatDataset
	transform = transforms.Compose([transforms.ToTensor()])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
								download=True, transform=transform)

	#stack all train images together into a tensor of shape 
	#(50000, 3, 32, 32)
	x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

	#get the mean of each channel            
	mean = torch.mean(x, dim=(0,2,3)) #tensor([0.4914, 0.4822, 0.4465])
	std = torch.std(x, dim=(0,2,3)) #tensor([0.2470, 0.2435, 0.2616])  
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import monai

from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

from monai.transforms import (
	Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
	Resized, ScaleIntensityd, RandFlipd, RandShiftIntensityd,
	RandScaleIntensityd, EnsureTyped, RandSimulateLowResolutiond,
	RandAdjustContrastd, RandRotate90d, RandZoomd, RandGaussianNoised,
	RandGaussianSmoothd, RandCoarseDropoutd, MapTransform
)


class ComputeSDFd(MapTransform):
	"""
	Transform to compute the Signed Distance Field (SDF) from the segmentation mask.
	This implements the level set representation phi_G described in the paper[cite: 152].
	"""

	def __init__(self, keys):
		super().__init__(keys)

	def __call__(self, data):
		d = dict(data)
		for key in self.keys:
			mask = d[key]
			# Ensure we are working with a numpy array for distance_transform_edt
			if isinstance(mask, torch.Tensor):
				mask_np = mask.detach().cpu().numpy()[0].astype(bool)
			else:
				mask_np = mask[0].astype(bool)

			if mask_np.any():
				negmask = ~mask_np
				# D_G(q) evaluates distance to the nearest point on boundary [cite: 141]
				dist_out = distance_transform_edt(negmask)
				dist_in = distance_transform_edt(mask_np)
				# phi_G(q) = -D_G(q) if q in G, and D_G(q) otherwise [cite: 152]
				sdf = dist_out - dist_in
			else:
				# For slices with only background, regional loss is sufficient [cite: 256]
				sdf = np.zeros_like(mask_np, dtype=np.float32)

			d["sdf"] = sdf[None, ...].astype(np.float32)
		return d


class ISICDatasetV2(Dataset):
	def __init__(self, dataframe, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		
		self.isic_ids = dataframe['isic_id'].values
		self.labels = dataframe['label'].astype(int).values
		
		self.image_paths = [os.path.join(self.root_dir, f"{isic_id}.jpg") 
							for isic_id in self.isic_ids]
		
		self.labels = torch.from_numpy(self.labels).long()

	def __len__(self):
		return len(self.isic_ids)

	def __getitem__(self, idx):
		img_path = self.image_paths[idx]
		label = self.labels[idx]
		isic_id = self.isic_ids[idx]

		image = Image.open(img_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image, label, isic_id
	
	@staticmethod
	def invert_img_norm(img):
		mean=[0.6689, 0.5090, 0.4417]
		std=[0.1336, 0.1352, 0.1486]
		inv_normalize = transforms.Normalize(
			mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
			std=[1 / std[0], 1 / std[1], 1 / std[2]]
		)
		img = inv_normalize(img)
		return transforms.ToPILImage()(img)
	

class ISICDatasetV1(Dataset):
	"""
	Custom PyTorch Dataset for the ISIC 2024 dataset structure.
	Reads image filenames and labels from the provided CSV.
	"""
	def __init__(self, img_dir, csv_file, transform=None):
		"""
		Args:
			img_dir (string): Directory with all the images.
			csv_file (string): Path to the CSV file with annotations.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.img_dir = img_dir
		self.transform = transform
		
		try:
			self.annotations = pd.read_csv(csv_file)
		except FileNotFoundError:
			print(f"Error: CSV file not found at {csv_file}")
			self.annotations = pd.DataFrame(columns=['isic_id', 'malignant']) 
	
		self.samples = []
		for _, row in self.annotations.iterrows():
			img_name = f"{row['isic_id']}.jpg"
			img_path = os.path.join(self.img_dir, img_name)
			
			if os.path.exists(img_path):
				self.samples.append((img_name, int(row['malignant'])))
			else:
				print(f"Warning: Image {img_name} listed in CSV but not found in {self.img_dir}")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name, label = self.samples[idx]
		img_path = os.path.join(self.img_dir, img_name)
		
		image = Image.open(img_path).convert('RGB')
		
		if self.transform:
			image = self.transform(image)

		return image, label


# Persisted lesion/patient-grouped split shared by every model in the workspace
# (built by splits/make_isic_clinical_split.py; override with "split_file" in the dataset config).
DEFAULT_SPLIT_FILE = Path(__file__).resolve().parents[1] / "splits" / "isic_clinical_v1.csv"


def load_split(split_file=None, data_dir=None):
	"""Read the persisted split; optionally check that every train/val/test image exists in data_dir.

	A relative split_file (e.g. "../splits/isic_clinical_v2.csv" in a config) is resolved against this
	folder, not the cwd: eval/export_predictions.py runs from the repo root and re-reads the same config.
	"""
	split_file = Path(split_file) if split_file else DEFAULT_SPLIT_FILE
	if not split_file.is_absolute():
		split_file = Path(__file__).resolve().parent / split_file
	df = pd.read_csv(split_file, dtype={"label": "Int64"})
	df = df[df["split"] != "excluded"].reset_index(drop=True)
	if data_dir is not None:
		missing = [i for i in df["isic_id"] if not os.path.exists(os.path.join(data_dir, f"{i}.jpg"))]
		if missing:
			raise FileNotFoundError(f"{len(missing)} images listed in {split_file} are missing from {data_dir} "
									f"(e.g. {missing[:3]}). Re-download the ISIC clinical close-ups.")
	return df


def get_isic_files(root_dir, subset):
	img_dir = os.path.join(root_dir, subset, "img")
	seg_dir = os.path.join(root_dir, subset, "seg")
	data_dicts = []
	for img_name in sorted(os.listdir(img_dir)):
		if not img_name.endswith(".jpg"): continue
		img_path = os.path.join(img_dir, img_name)
		seg_path = os.path.join(seg_dir, img_name.replace(".jpg", "_segmentation.png"))
		if os.path.exists(seg_path):
			data_dicts.append({"img": img_path, "seg": seg_path, "is_pseudo": False, "loss_weight": 1.0})
	return data_dicts


def subsample_train(df_train, fraction, seed=0):
	"""Keep `fraction` of the training lesion/patient groups, stratified by label (for learning curves).
	Whole groups are kept or dropped, so no group is split; val/test are never touched."""
	groups = df_train.groupby("group_id")["label"].max().reset_index()
	keep = (groups.groupby("label", group_keys=False)
			.apply(lambda g: g.sample(frac=fraction, random_state=seed))["group_id"])
	return df_train[df_train["group_id"].isin(set(keep))].reset_index(drop=True)


def medirv2_transforms(input_size):
	"""(train, test) transforms for MEDIRV2; shared with evaluation scripts so preprocessing never drifts."""
	normalize = transforms.Normalize(mean=[0.6689, 0.5090, 0.4417],
									std=[0.1336, 0.1352, 0.1486])
	train_transform = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomRotation(60),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
		transforms.ToTensor(),
		normalize,
	])
	test_transform = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		normalize,
	])
	return train_transform, test_transform


def get_dataloader(root: str, input_size: int=0, data_config: dict={}, test: bool=False, generator=None):

	dataset = data_config["dataset"]
	train_shuffle = True if data_config["kfold"] < 2 else False
	
	if dataset.upper() == "MEDIRV1":

		### Dataset used by Tyson, includes ISIC_2024_Permissive_Training_Input + Melanoma Cancer Dataset ###
		
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])

		train_transform = transforms.Compose([
			transforms.Resize((224, 224)),  
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(20),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
			transforms.ToTensor(),
			normalize,
		])

		test_transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			normalize,
		])

		isic_img_dir = os.path.join(root, 'ISIC_2024_Permissive_Training_Input')
		isic_csv_file = os.path.join(isic_img_dir, 'ISIC_2024_Permissive_Training_GroundTruth.csv')
		
		isic_dataset = ISICDatasetV1(
			img_dir=isic_img_dir,
			csv_file=isic_csv_file,
			transform=train_transform
		)

		melanoma_train_dir = os.path.join(root, 'melanoma_cancer_dataset', 'train')
		melanoma_train_dataset = ImageFolder(
			root=melanoma_train_dir,
			transform=train_transform
		)
		
		train_data = ConcatDataset([isic_dataset, melanoma_train_dataset])

		if data_config["run_val"] or test: 
			melanoma_test_dir = os.path.join(root, 'melanoma_cancer_dataset', 'test')
			test_data = ImageFolder(root=melanoma_test_dir, transform=test_transform)

	
	elif dataset.upper() == "MEDIRV2":
		
		data_dir = os.path.join(root, "ISIC_clinical")
		# Lesion/patient-grouped, persisted split (Indeterminate/unlabelled images already excluded).
		# Replaces the old per-run image-level split, which put ~26% of test images in the same
		# lesion/patient group as a training image (see splits/isic_clinical_v1_report.md).
		df = load_split(data_config.get("split_file"), data_dir)

		train_transform, test_transform = medirv2_transforms(input_size)

		if data_config["kfold"] != 1:
			raise NotImplementedError("MEDIRV2 uses the persisted split file; k-fold is not supported (set kfold=1).")
		else:
			df_train = df[df["split"] == "train"].reset_index(drop=True)
			if data_config.get("train_fraction", 1.0) < 1.0:
				df_train = subsample_train(df_train, data_config["train_fraction"], data_config.get("subsample_seed", 0))
			df_val = df[df["split"] == "val"].reset_index(drop=True)
			df_test = df[df["split"] == "test"].reset_index(drop=True)

			if test:
				test_data = ISICDatasetV2(df_test, data_dir, transform=test_transform)
			else:
				train_data = ISICDatasetV2(df_train, data_dir, transform=train_transform)
				test_data = ISICDatasetV2(df_val, data_dir, transform=test_transform)

	elif dataset.upper() == "ISIC2018":

		### ISIC Segmentation data from : https://challenge.isic-archive.com/landing/2018/
		train_files = get_isic_files(root, "ISIC2018/train")
		if test:
			test_files = get_isic_files(root, "ISIC2018/test")
		else: 
			test_files = get_isic_files(root, "ISIC2018/val")

		IMAGENET_MEAN = [0.485, 0.456, 0.406]
		IMAGENET_STD = [0.229, 0.224, 0.225]

		# Base pipeline to pre-calculate SDF once [cite: 81, 160]
		base_transforms = [
			LoadImaged(keys=["img", "seg"]),
			EnsureChannelFirstd(keys=["img", "seg"]),
			ScaleIntensityd(keys=["img", "seg"], minv=0.0, maxv=1.0),
			Resized(keys=["img", "seg"], spatial_size=(256, 256), mode=["bilinear", "nearest"]),
			ComputeSDFd(keys=["seg"]),
		]

		train_transforms = Compose(base_transforms + [
			RandRotate90d(keys=["img", "seg", "sdf"], prob=0.5),
			RandFlipd(keys=["img", "seg", "sdf"], prob=0.5),
			RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
			RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.5),
			RandAdjustContrastd(keys=["img"], prob=0.3, gamma=(0.7, 1.5)),
			NormalizeIntensityd(keys=["img"], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
			EnsureTyped(keys=["img", "seg", "sdf"]),
		])

		test_transforms = Compose(base_transforms + [
			NormalizeIntensityd(keys=["img"], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
			EnsureTyped(keys=["img", "seg", "sdf"]),
		])

		train_data = monai.data.PersistentDataset(train_files, train_transforms, cache_dir='../cache/')
		test_data = monai.data.PersistentDataset(test_files, test_transforms, cache_dir='../cache/')

	else:
		raise Exception(f"Dataset {dataset}' has no dataloader implemented") 

	if not test: 
		if data_config["kfold"] > 1: 
			return train_data, None 
		
		train_loader = DataLoader(dataset=train_data, batch_size=data_config["batch_size"],
							shuffle=train_shuffle, pin_memory=True, drop_last=True,
							generator=generator if train_shuffle else None)
	
	if data_config["run_val"] or test: 
		if data_config["kfold"] > 1 and test: 
			return train_data
		test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=True)
		if test:
			return test_loader
		else:
			return train_loader, test_loader