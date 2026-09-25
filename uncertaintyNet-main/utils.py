import csv, os, json, glob, math, h5py, operator
from pathlib import Path
import torch
import numpy as np
from collections.abc import Iterable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""==================================================="""
"""================ Metrics Utilities ================"""
"""==================================================="""
def get_one_hot(targets: torch.Tensor, num_classes: int, dataset: str=""):
	"""
	Converts an integer label torch.autograd.Variable to a one-hot Variable.

	Args:
		targets (torch.autograd.Variable): A tensor of shape (N, 1) containing labels.
		num_classes (int): The number of classes in the labels.

	Returns:
		torch.FloatTensor: A tensor of shape (N, num_classes) with labels in one-hot format.
	"""
	one_hot = F.one_hot(targets.to(torch.long), num_classes=num_classes).float()
	if dataset.upper() == 'ISIC2018':
		return one_hot.permute(0, 4, 2, 3, 1).squeeze(-1)
	if dataset.upper() == 'DERMAMNIST':
		return one_hot.squeeze(1)
	return one_hot

def calculate_iou(input: torch.Tensor, target: torch.Tensor, num_classes: int, dataset: str=''):
	"""
	Calculate Intersection over Union (IoU) for multi-class classification using PyTorch.

	Parameters:
		target (torch.Tensor): Ground truth labels (2D tensor).
		input (torch.Tensor): Predicted softmax output (2D tensor).
		num_classes (int): Number of classes.

	Returns:
		float: Mean IoU across all classes.
	"""
	iou_scores = []

	dim = 2 if len(input.shape) > 4 else 1
	pred = torch.argmax(input, dim=dim)
	if len(target.shape) > 3: 
		target = torch.argmax(target.clone(), dim=1)
	
	if dim == 2: 
		target = target.expand_as(pred)

	if dataset == 'cityscapes':
		void_class = (target == 0).bool()
	for class_id in torch.unique(target):
		if class_id == 0:
			#Need to remove void classes from the calculation
			continue
		#ignore class index 0 as it is associated with void classes/background
		true_class = (target == class_id).float()
		pred_class = (pred == class_id).float()

		intersection = torch.logical_and(true_class, pred_class).float()
		union = torch.logical_or(true_class, pred_class).float()

		iou = intersection / union
		if dataset == 'cityscapes':
			iou = torch.nanmean(iou[~void_class])
		else:
			iou = torch.nanmean(iou)
		iou_scores.append(iou.item())

	mean_iou = np.nanmean(iou_scores) #ignore nan values (empty ground truth classes)
	return mean_iou


def calculate_dice(input: torch.Tensor, target: torch.Tensor, num_classes: int, epsilon=1e-5):
	"""
	Compute the Dice coefficient for multi-class segmentation.

	Args:
		input (torch.Tensor): Predicted tensor with shape (batch_size, num_classes, height, width).
		target (torch.Tensor): Ground truth tensor with shape (batch_size, num_classes, height, width).
		num_classes (int): Number of classes.
		epsilon (float): Smoothing term to avoid division by zero.

	Returns:
		float: Mean Dice coefficient value across all classes.
	"""
	targets_ = target.expand_as(input)
	intersection = input * targets_ 
	union = input + targets_
	dice = (2.0 * intersection + epsilon) / (union + epsilon)
	return dice.mean().item()

def compute_brier_score(logits: torch.Tensor, targets: torch.Tensor, 
						nll_loss: bool=False, one_hot_targets: bool=False,
						num_classes: int=2, task: str='classification', 
						multi_label: bool=False, dataset: str=''):
	"""
	Compute the Brier score.

	Args:
	- logits (torch.Tensor): Raw scores from the model.
	- targets (torch.Tensor): Ground truth labels.
	- nll_loss (bool): whether log_softmax layer is already applied to logits

	Returns:
	- float: Brier score.
	"""
	if multi_label: 
		probs = F.sigmoid(logits)
		targets_ = targets
	else:
		if task == 'classification' or task == 'segmentation':
			# Apply softmax to obtain probabilities
			if nll_loss:
				probs = torch.exp(logits)
			else:
				dim = -1 if task == 'classification' else -3
				probs = F.softmax(logits, dim=dim)
				
		# One-hot encode the targets
		if (not one_hot_targets and task == 'classification') or (not one_hot_targets and dataset == 'cityscapes'): 
			targets_ = F.one_hot(targets, num_classes=num_classes)
			if dataset == 'cityscapes':
				targets_ = targets_.permute(0, 3, 1, 2)
		else: 
			targets_ = targets
	targets_ = targets_.expand_as(probs)
	# Compute Brier score
	class_dim = -1 if task == 'classification' else -3
	brier_score = torch.sum((probs - targets_.float())**2, dim=class_dim) / num_classes

	return torch.mean(brier_score).item()

def expected_calibration_error(target, y_prob, num_bins=10):
	"""
	Compute the Expected Calibration Error (ECE) for a classification model.

	Parameters:
	target (array-like): True labels (0 or 1).
	y_prob (array-like): input probabilities for the positive class.
	num_bins (int): Number of bins to divide the data into for calibration.

	Returns:
	ece (float): The Expected Calibration Error.
	"""
	# Ensure target and y_prob have the same length
	if len(target) != len(y_prob):
		raise ValueError("Input arrays must have the same length.")

	# Initialize variables to store ECE and total number of samples
	ece = 0.0
	total_samples = len(target)

	# Calculate bin edges
	bin_edges = np.linspace(0, 1, num_bins + 1)

	# Calculate bin widths
	bin_widths = np.diff(bin_edges)

	for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:]):
		# Filter samples falling within the current bin
		in_bin = (y_prob >= bin_start) & (y_prob < bin_end)

		# Number of samples in the current bin
		num_samples_in_bin = np.sum(in_bin)

		if num_samples_in_bin > 0:
			# Calculate the average input probability in the bin
			avg_input_prob = np.mean(y_prob[in_bin])

			# Calculate the average true probability in the bin
			avg_true_prob = np.mean(target[in_bin])

			# Calculate the absolute difference between input and true probabilities
			ece += (num_samples_in_bin / total_samples) * np.abs(avg_input_prob - avg_true_prob)

	return ece


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor,
						nll_loss: bool=False, one_hot_targets: bool=False, 
						multi_label: bool=False, topk: int=1):
	if multi_label:
		probs = F.sigmoid(output)
		pred_binary = probs > 0.5
		if pred_binary.shape != target.shape:
			pred_binary = torch.mode(pred_binary, dim=0).values
		accuracy = (pred_binary == target).float().mean()
	else: 
		if nll_loss: 
			probs = torch.exp(output, dim=-1)
		else:
			probs = F.softmax(output, dim=-1)

		if one_hot_targets:
			target_ = torch.argmax(target, dim=1)
		else: 
			target_ = target
		
		_, topk_pred = torch.topk(probs, topk, dim=-1)

		targetk = target_.unsqueeze(1).expand_as(topk_pred)
		pred_class = (targetk == topk_pred).sum(-1)

		if pred_class.shape != target_.shape:
			accuracy = torch.mode(pred_class, dim=0).values.float()
		else:
			accuracy = pred_class.float()
			
	return accuracy.mean().item()

def entropy_of_expected(probs, epsilon=1e-10, class_dim=-1):
	"""
	param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
	dim: class dimension
	return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
	"""
	mean_probs = np.mean(probs, axis=0)
	log_probs = -np.log(mean_probs + epsilon)
	return np.sum(mean_probs * log_probs, axis=class_dim)

def expected_entropy(probs, epsilon=1e-10, class_dim=-1):
	"""
	:param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
	:return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
	"""
	log_probs = -np.log(probs + epsilon)
	return np.mean(np.sum(probs * log_probs, axis=class_dim), axis=0)

def predictive_entropy(logits, nll_loss=False, task='', multi_label=False):
	if task != 'segmentation':
		if nll_loss: 
			probs = torch.exp(logits, dim=-1)
		else:
			if multi_label: 
				probs = F.sigmoid(logits)
			else:
				probs = F.softmax(logits, dim=-1)
	else:
		probs = logits
	dim = -3 if task == 'segmentation' else -1
	eoe = entropy_of_expected(probs.detach().cpu().numpy(), class_dim=dim)
	ee = expected_entropy(probs.detach().cpu().numpy(), class_dim=dim)
	return eoe.mean(), ee.mean()
		
"""==================================================="""
"""================ Network Utilities ================"""
"""==================================================="""
class DiceLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		return 1-calculate_dice(input, target)

def get_nonlinearity(nonlinearity:str):
	if nonlinearity == "sigmoid":
		return nn.Sigmoid()
	elif nonlinearity == "relu":
		return nn.ReLU(inplace=True)
	elif nonlinearity == "softplus":
		return nn.Softplus()
	elif nonlinearity == 'tanh':
		return nn.Tanh()
	elif nonlinearity == 'leaky_relu':
		return nn.LeakyReLU()
	else: 
		raise Exception(f"Nonlinearity '{nonlinearity}' not found")

def load_checkpoint(net:nn.Module, init_path: str, cont_run = False, fold=None):
	try:
		print(f"Network initializing with parameters from {init_path}")
		init_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'output', init_path))
	except:
		raise Exception(f'Error when loading initialization weights! Please make sure that weights exist at {init_path}!')

	if cont_run:
		if fold is not None:
			file = init_path+f'/checkpoint_fold{fold}.pth.tar'
		else:
			file = init_path+'/checkpoint.pth.tar'
	elif os.path.isdir(init_path):
		os.chdir(init_path)
		if fold is not None:
			list_checkpoints = glob.glob(f"checkpoint_best_*fold{fold}*")
		else:
			list_checkpoints = glob.glob('checkpoint_best_*') 

		if len(list_checkpoints) == 0:
			print("No best performance checkpoint found. Will load any checkpoint saved")
			if fold is not None: 
				list_checkpoints = glob.glob(f"checkpoint_*fold{fold}*")
				file = max(list_checkpoints, key=os.path.getctime)
			else:
				file = glob.glob("*pth")
				if file == []:
					file = glob.glob("*.ckpt")[0]
				else:
					file = file[0]

		else:
			file = max(list_checkpoints, key=os.path.getctime)
	else: 
		file = init_path
	# try:
	print(f"Loading checkpoint from {file}")
	checkpoint = torch.load(file, weights_only=False)
	try:
		net.load_state_dict(checkpoint['network_state_dict'])
	except:
		if 'init_network' in checkpoint.keys():
			net = checkpoint['init_network']
			net.load_state_dict(checkpoint['network_state_dict'])
	epoch_start =0
	if cont_run:
		epoch_start = checkpoint['epoch']
	return net, epoch_start


def get_loss_func(loss_func: str='CE', label_smoothing: float=0, dataset:str=""):
	if loss_func.upper() == 'MSE':
		return nn.MSELoss()
	elif loss_func.upper() == 'CE':
		return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
	elif loss_func.upper() == 'BCE':
		return nn.BCELoss()
	elif loss_func.upper() == 'BCELOGITS':
		return nn.BCEWithLogitsLoss()
	elif loss_func.upper() == 'NLL':
		return nn.NLLLoss()
	elif loss_func.upper() == 'DICE':
		return DiceLoss()
	else: 
		raise Exception(f"{loss_func} LOSS FUNCTION NOT IMPLEMENTED")

def get_lr_lambda(warmup_epochs, initial_lr, max_lr, drop_epochs, drop_factor):
	def lr_lambda(current_epoch):
		if current_epoch < warmup_epochs:
			return (max_lr - initial_lr) / warmup_epochs * current_epoch + initial_lr
		else:
			if current_epoch >= max(drop_epochs):
				lr = max_lr*drop_factor**len(drop_epochs)
			else: 
				for i, epoch in enumerate(drop_epochs):
					epoch_i = warmup_epochs if i ==0 else drop_epochs[i-1]
					if current_epoch in range(epoch_i, epoch):
						lr = max_lr*(drop_factor**i)
						break
			return lr
	return lr_lambda

def get_lr_scheduler(scheduler_info: dict, optimizer:torch.optim):
	scheduler_type = scheduler_info["scheduler_type"]
	try:
		gamma = scheduler_info["gamma"]
		if scheduler_type == "": 
			return None 
		elif scheduler_type.upper() == "LAMBDA":
			# Example usage:
			warmup_epochs = scheduler_info["step_size"][0]
			min_lr = 0.001
			initial_lr = 0.1  # Low initial learning rate
			max_lr = 1.6
			drop_epochs = scheduler_info["step_size"][1:]
			drop_factor = gamma

			lr_lambda = get_lr_lambda(warmup_epochs, initial_lr, max_lr, drop_epochs, drop_factor)
			return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
		elif scheduler_type.upper() == "STEP":
			step_size = scheduler_info["step_size"]
			if isinstance(step_size, list):
				return torch.optim.lr_scheduler.MultiStepLR(optimizer, step_size, gamma)
			return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
		elif scheduler_type.upper() == "REDUCELRONPLATEAU":
			return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=gamma, min_lr=1e-6)
		elif scheduler_type.upper() == "COSINE":
			return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_info["T_max"], eta_min=scheduler_info["min_lr"])
	except:
		raise Exception(f"{scheduler_type} Scheduler is not implemented, \
				  choose one of the following options: ['', step, ReduceLROnPlateau]")

def get_optimizer(model, opt_type='adam', lr=0.1, weight_decay=0, sensitivity=None):
	if opt_type == 'adam':
		return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif opt_type == 'sgd':
		if torch.__version__[0] == '2':
			return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, foreach=False, weight_decay=weight_decay)
		else: 
			return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
	elif opt_type == 'adagrad':
		return torch.optim.Adagrad(model.parameters(), lr=lr, foreach=False, weight_decay=weight_decay)
	else: 
		raise Exception(f"{opt_type} OPTIMIZER NOT IMPLEMENTED")

def gimme_params(model: nn.Module, partial_bayes: bool=False):
	if partial_bayes:
		params, total = 0, 0
		for name, buffer in model.named_buffers():
			if 'mask' in name and buffer is not None: 
				#partial bayes params
				params += (~buffer).sum() 
		for name, param in model.named_parameters():
			if 'bn' not in name:
				total += param.numel()
		params = params
		total = total
	else:
		model_parameters = [p for p in model.parameters() if p.requires_grad]
		params = sum([p.numel() for p in model_parameters if not p.is_sparse])
		params += sum([p._nnz() for p in model_parameters if p.is_sparse])
		total = sum([p.numel() for p in model_parameters])
	return params, total

"""==================================================="""
"""================ General Utilities ================"""
"""==================================================="""
def get_logging_keys(net, task, run_val=False, test=False, dataset=''):
	keys_mapping = {
		"deterministic": {
			"regression": ["RMSE"],
			"classification": ["Accuracy"],
			"segmentation": ["Dice"],
			"segmentation2": ["IoU"],
		},
		"bayesian": {
			"regression": ["KL", "RMSE", "Std"],
			"classification": ["KL", "Accuracy", "Brier Score"],
			"segmentation": ["KL", "Dice", "Brier Score"],
			"segmentation2": ["KL", "IoU", "Brier Score"],
		},
		"partial_bayesian": {
			"regression": ["KL", "RMSE", "Std"],
			"classification": ["KL", "Accuracy", "Brier Score"],
			"segmentation": ["KL", "Dice", "Brier Score"],
			"segmentation2": ["KL", "IoU", "Brier Score"],
		}
	}

	if dataset == "cityscapes" or dataset == 'isic':
		task = "segmentation2"

	if isinstance(task, list):
		keys = []
		for task_ in task:
			for key in keys_mapping.get(net, {}).get(task_, []):
				keys.append(f"Train {key}")
		train_keys = list(set(keys))
	else:
		train_keys = [f"Train {key}" for key in keys_mapping.get(net, {}).get(task, [])]
	test_keys = []
	if run_val or test:
		keys_suffix = "Test" if test else "Val"
		test_keys = [f"{keys_suffix} Time", f"{keys_suffix} Loss"] if keys_suffix == 'Val' else [f"{keys_suffix} Time"]
		if isinstance(task, list):
			keys = []
			for task_ in task:
				for key in keys_mapping.get(net, {}).get(task_, []):
					keys.append(f"{keys_suffix} {key}")
			test_keys += list(set(keys))
		else:
			test_keys += [f"{keys_suffix} {key}" for key in keys_mapping.get(net, {}).get(task, [])]

	if test and "segmentation" not in task:
		test_keys.append("Test NLL")
		test_keys.append("Test Expected Entropy")
		test_keys.append("MaxSoftmax")
		if "bayesian" in net:
			test_keys.append("VarSoftmax")
		if task == "regression":
			logging_keys = ["Y", "Y_pred"] + test_keys
		elif task == "classification":
			if "CHESTMNIST" in dataset.upper():
				logging_keys = ["Y_conf"] + test_keys + ["Test Entropy", "TP", "TN", "FP", "FN", "Test Brier Score"]
			elif "CIFAR100" in dataset.upper() or "IMAGENET" in dataset.upper():
				logging_keys = ["Y", "Y_pred", "Y_conf"] + test_keys + ["Test Entropy", "Test Brier Score", "Top5 Accuracy"]
			elif dataset.upper() == "HALFMOON":
				logging_keys = ["X1", "X2", "Y", "Y_pred", "Y_conf"] + test_keys + ["Test Entropy", "Test Brier Score"]
			else:
				logging_keys = ["Y", "Y_pred", "Y_conf"] + test_keys + ["Test Entropy", "Test Brier Score"]
	elif test and "segmentation" in task:
		test_keys.append("Test NLL")
		test_keys.append("Test Expected Entropy")
		logging_keys = test_keys + ["Test Entropy"]
		if "Test Brier Score" not in logging_keys:
			logging_keys.append("Test Brier Score")
	else:
		logging_keys = ["Kfold", "Epoch", "Train Time", "Train Loss"] + train_keys + test_keys

	metrics = {key: [] for key in logging_keys}

	if (run_val or not test) and dataset != '':
		best_metric_prefix = "Best Val" if run_val else "Best Train"
		if task == "classification":
			metrics[f'{best_metric_prefix} Accuracy'] = 0
		elif task == 'regression':
			metrics[f'{best_metric_prefix} RMSE'] = 1e4
		elif task == 'segmentation':
			metrics[f'{best_metric_prefix} Dice'] = 0
		if task == 'segmentation2':
			metrics[f'{best_metric_prefix} IoU'] = 0
	else: 
		metrics["Best Val Loss"] = 1e4

	return metrics, logging_keys


class CSVlogger():
	def __init__(self, log_name, header_names):
		self.logging_keys = header_names
		self.log_name     = log_name
	   
		with open(log_name,"a") as csv_file:
			writer = csv.writer(csv_file, delimiter=",")
			writer.writerow(self.logging_keys)
	def write(self, inputs):
		with open(self.log_name,"a") as csv_file:
			writer = csv.writer(csv_file, delimiter=",")
			writer.writerow(inputs)

def read_config(out_dir: str, file='config.json'):
	file = Path(out_dir) / file
	with open(file, 'r') as f:
		opts = json.load(f)
	return opts

def log_runs(opts: dict, out_dir = None, file='log_runs.csv'):

	run = opts["Paths"]["save_path"]
	log_keys = ["run"]
	log_values = [Path(run).resolve().name]

	if out_dir is None: 
		out_dir = Path(__file__).resolve().parent
	
	opts = parse_json(opts, net_type=opts["Network"]["Basic Setup"]["network_type"])
	log_keys += list(opts.keys())    
	log_values += list(opts.values()) 

	log_file_path = Path(out_dir) / file

	if not log_file_path.exists():
		with open(log_file_path, 'w') as log_file:
			log_file.write(','.join(log_keys))

	with open(log_file_path, 'a') as log_file:
		log_file.write("\n")
		for val in log_values:
			log_file.write(str(val) + ',')
		

def parse_json(data: json, keys={}, net_type="deterministic", empty=False):
	empty_values = False
	if "partial" not in net_type: 
		empty_values = True

	for key, value in data.items(): 
		if isinstance(value, dict):
			if "partial" in key and empty_values: 
				empty = True 
			parse_json(value, keys=keys, empty=empty)
		else:
			if empty == True: 
				value = "" 
			keys.update({key: value})
	return keys

def save_pred(input, target, output, epoch, output_path, dataset, train=True, *args):
	num_samples = 5 if train else 1
	tar_idx = 0 if target.shape[1] == 1 else 1
	posterior = False
	cols = 5 if args else 4
	origin = 'upper' if dataset == 'chestX' else 'lower'

	if output.shape != target.shape:
		posterior = True
		variance = output.var(dim=0)
		output = output.mean(dim=0)
	cols += 1 if posterior else 0
	
	fig, axarr = plt.subplots(num_samples, cols, figsize=(cols * 2, num_samples * 2), gridspec_kw={'wspace': 0.02, 'hspace': 0})
	axarr = axarr.flatten()
	
	for i in range(num_samples):
		img = input[i].squeeze(0).cpu().numpy()
		cmap_in, cmap = 'gray', 'gray'

		label = target[i][tar_idx].detach().cpu().numpy()
		conf = output[i][tar_idx].detach().cpu().numpy()
		pred = output[i][tar_idx].detach().cpu().numpy() > 0.5
		cmap = 'gray'
		
		indices = [i * cols + j for j in range(cols)]
		axarr[indices[0]].imshow(img, cmap=cmap_in, origin=origin)
		axarr[indices[1]].imshow(label, cmap=cmap, origin=origin)
		axarr[indices[2]].imshow(pred, cmap=cmap, origin=origin)
		
		if i == 0:
			axarr[indices[0]].title.set_text('input')
			axarr[indices[1]].title.set_text('target')
			axarr[indices[2]].title.set_text('pred')
		
		if cols > 3:
			axarr[indices[3]].imshow(conf, cmap='gray', origin=origin)
			if args:
				uncert_gt = args[0][i].detach().squeeze(0).cpu().numpy()
				axarr[indices[4]].imshow(uncert_gt, cmap='seismic', origin=origin)
			
			if i == 0:
				axarr[indices[3]].title.set_text('conf')
				if args and cols == 5:
					axarr[indices[4]].title.set_text('gt_u')
			
			if posterior:
				uncert_pred = variance[i][tar_idx].detach().squeeze(0).cpu().numpy()
				axarr[indices[-1]].imshow(uncert_pred, cmap='seismic', origin=origin)
				if i == 0:
					axarr[indices[-1]].title.set_text('pred_u')
				

	for i in range(num_samples):
		for j in range(cols): 
			idx = (i, j) if num_samples != 1 else j
			axarr[idx].axis('off') 
	os.chdir(output_path)
	plt.savefig(f'pred{epoch}.pdf', format="pdf", dpi=1200)
	plt.close('all')

def append_to_hdf5(file_name, output_path, name, img, target, pred, *args):
	name = str(name)
	file_path = os.path.join(output_path, file_name)
	with h5py.File(file_path, 'a') as hf:
		group = hf.create_group(name)
		group.create_dataset('image', data=img.squeeze(0).squeeze(0).detach().cpu())
		group.create_dataset('mask', data=target.squeeze(0).detach().cpu())
		group.create_dataset('pred', data=pred.detach().cpu())
		if args:
			group.create_dataset('gt_uncert', data=args[0].detach().cpu().squeeze(0).squeeze(0))
			group.create_dataset('gt_mean', data=args[1].detach().cpu().squeeze(0).squeeze(0))
			group.create_dataset('masks', data=args[2].detach().cpu())

