import argparse
import os, operator
from pathlib import Path
import numpy as np
import random, json
import dataset
import models.model as model
from sklearn.model_selection import KFold
import time, copy
import torch
import torch.utils.data
from tqdm import tqdm, trange
import utils
import models.model_utils as mutils
import models.bayes_models as bayesmodels

from lightning.fabric import Fabric

#######################################################################################################
### ARGUMENTS
parse_in = argparse.ArgumentParser()
parse_in.add_argument('--config',   type=str, default='',
										help='Path to configuration file')
parse_in.add_argument('--search_setup', type=str, default='',
										help='Path to search for config file')
parse_in.add_argument('--no_date',      action='store_true', help='Do not use date when logging files.')
parse_in.add_argument('--continue_run', action='store_true', help='continue interrupted run')
parse_in.add_argument('--seed', 		type=int, default=None, help='random seed' )
parse_in.add_argument('--multi_gpu',	action='store_true', default=False, help='Use multiple GPUs if available.')

opts = parse_in.parse_args()

assert opts.config!='', 'Please provide a config file'

if opts.search_setup == '': 
	opts.search_setup = Path(__file__).resolve().parent

config = utils.read_config(opts.search_setup/'configs', file=opts.config)
if config["Paths"]["save_path"] == '':
	config["Paths"]["save_path"] = str(Path(__file__).resolve().parent / 'output' / 'run')
else:
	config["Paths"]["save_path"] = os.path.join(os.path.abspath(config["Paths"]["save_path"]), 'run')
if not opts.no_date:
	config["Paths"]["save_path"] += '_' + time.strftime("%Y%m%d-%H%M%S")
	
if config["Paths"]["dataset_path"] != '':
	dataset_path = config["Paths"]["dataset_path"]
else:
	dataset_path = os.path.join(opts.search_setup, 'datasets')

### LOAD CONFIGS
net_config = config["Network"]["Basic Setup"]
data_config = config["Training"]["Dataset"]
train_config = config["Training"]["Parameters"]
bayes_config = config["Training"]["Bayesian Parameters"]
pbayes_config = config["Training"]["Partial Bayesian Parameters"]
lbayes_config = config["Training"]["Layer Bayesian Parameters"]
criterion = utils.get_loss_func(train_config["loss_func"], label_smoothing=train_config["label_smoothing"], dataset=data_config["dataset"]) 
nll_loss = True if train_config["loss_func"] == "nll_loss" else False

#######################################################################################################
### SET UP DEVICE AND MULTI-GPU IF NEEDED
multi_gpu = False
fabric=None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opts.multi_gpu:
	# if net_config["network_type"] == "partial_bayesian":
		# assert opts.pbayes_run is not None, "Please provide the path to the initialization network for Partial Bayesian multi-GPU training using --pbayes_run"
	multi_gpu = True
	fabric = Fabric(accelerator="cuda", devices=[0,1], strategy="ddp")
	fabric.launch()

#######################################################################################################
### LOAD DATA
data_loader, val_loader = dataset.get_dataloader(root=dataset_path, input_size=net_config["input_size"], data_config=data_config)

#######################################################################################################
### NETWORK INITIALIZATION
def initialize_partial_bayes(fold=None):
	pbayes_config = config["Training"]["Partial Bayesian Parameters"]
	init_net = pbayes_config["init_network"]
	if init_net == '' or "MNIST" in init_net:
		print("No init network provided for Partial Bayesian network")
		print("Attempting to load network based on config provided")
		init_net_config = copy.deepcopy(config)
		init_net_config['Network']['Basic Setup']['network_type'] = 'deterministic'
	else:
		init_net_path = Path(config["Paths"]["save_path"]).parent / init_net
		if not init_net_path.exists(): 
			output_directory = next((p for p in reversed(Path(config["Paths"]["save_path"]).parents) if p.name == 'output'), None)
			init_net_path = output_directory / init_net
			if not init_net_path.exists():
				raise Exception(f"Initialization Network '{init_net}' is either not a correct path or not an existing run")
		init_net_config = utils.read_config(Path(__file__).resolve().parent/'output'/init_net)

	init_net_setup = init_net_config['Network']['Basic Setup']
	opts.seed = init_net_config['Training']['Parameters']['seed']

	### LOAD THE INITIALIZATION NETWORK 
	network = model.get_model(init_net_setup)
	if init_net != '':
		network, _ = utils.load_checkpoint(net=network, init_path=init_net, fold=fold, cont_run=False)

	network = mutils.build_variational_model(network, net_config["network_type"], config["Paths"]["save_path"], 
								bayes_config, pbayes_config, lbayes_config, multi_gpu=multi_gpu, fabric=fabric)
	return network

if net_config["network_type"] == "partial_bayesian":
	fold = 0 if data_config["kfold"] > 1 else None 
	network = initialize_partial_bayes(fold)
else: 
	if net_config["network_type"] == "bayesian":
		init_net_config = copy.deepcopy(config)
		init_net_config['Network']['Basic Setup']['network_type'] = 'deterministic'
		init_net_setup = init_net_config['Network']['Basic Setup']
		network = model.get_model(init_net_setup)
		network = mutils.build_variational_model(network, net_config["network_type"], config["Paths"]["save_path"], bayes_configs, pbayes_config, lbayes_config)
	else:
		network = model.get_model(net_config)

if "bayesian" in net_config["network_type"]:
	kl_weight = bayes_config["kl_weight"]
	m_val_batch, len_val_data = None, None 
	if data_config["run_val"] and data_config["kfold"] == 1:
		m_val_batch=len(val_loader)
		len_val_data=len(val_loader.dataset)
	len_train_data = len(data_loader) if data_config["kfold"] > 1 else len(data_loader.dataset)

	kl_weight = mutils.KL_Weight(model = network, 
								m_train_batch=len(data_loader), m_val_batch=m_val_batch, 
								len_train_data=len_train_data, len_val_data=len_val_data,
								method=kl_weight, num_epochs=train_config["num_epochs"],
								hi2low=bayes_config["kl_weight_hi2low"], 
								kl_weight_lim=bayes_config["kl_weight_lim"])
								
if train_config["continue_run"]: 
	network, epoch_start = utils.load_checkpoint(network, train_config["init_network"], cont_run=True) #continue_run=True to load the last checkpoint
else:
	epoch_start = 0

#######################################################################################################
if opts.seed is not None:
	seed = opts.seed
	train_config['seed'] = seed
else:
	seed = train_config['seed']

### SET UP SEED FOR REPRODUCIBILITY
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#######################################################################################################
## LOGGING
if not os.path.isdir(config["Paths"]['save_path']):
	if (multi_gpu and fabric.global_rank == 0) or not multi_gpu:
		os.makedirs(config["Paths"]['save_path'])

if (multi_gpu and fabric.global_rank == 0) or not multi_gpu:
	os.chdir(config["Paths"]['save_path'])
	with open('config.json', 'w') as fp:
		json.dump(config, fp)

metrics, logging_keys = utils.get_logging_keys(net_config["network_type"], net_config["task"], 
											   data_config["run_val"], dataset=data_config["dataset"]) 
full_log = None
if (multi_gpu and fabric.global_rank == 0) or not multi_gpu:
	full_log  = utils.CSVlogger(Path(config["Paths"]['save_path'])/"log.csv", logging_keys)

network.to(device)
network.n_params = utils.gimme_params(network, partial_bayes=True if net_config["network_type"] == 'partial_bayesian' else False)

#######################################################################################################
## COMPILE MODEL FOR FASTER TRAINING (PYTORCH 2.0+)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
compiled = False
if hasattr(torch, 'compile'):
	print("Compiling model (this may take a minute at start)...")
	# 'reduce-overhead' uses CUDA graphs which is super fast for small batches (CIFAR)
	try:
		network = torch.compile(network, mode='reduce-overhead')
	except Exception as e:
		print(f"Compilation with reduce-overhead failed, falling back to default: {e}")
		network = torch.compile(network)
	compiled = True
	print("Model compiled.")	

#######################################################################################################
## OPTIMIZATION SETUP
optimizer = utils.get_optimizer(network, opt_type=train_config["optimizer"], lr=train_config["lr"], weight_decay=train_config["weight_decay"])

lr_scheduler = utils.get_lr_scheduler(config["Training"]["LR Scheduler"], optimizer)
if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau" and not data_config["run_val"]:
	raise Exception("If ReduceLROnPlateau is selected as a scheduler, run_val flag cannot be false")
#######################################################################################################
### START TRAINING
full_training_start_time = time.time()

def train_single_fold(network, train_loader, val_loader, optimizer, output_path, full_log, fold=0, network_type='', multi_gpu=False):
	epoch_iter = trange(epoch_start, train_config["num_epochs"], position=1)
	for epoch in epoch_iter:
		network.train()
		epoch_time = time.time()
		train_data_iter = tqdm(train_loader, position=2)
		loss = 0
		total_loss = 0
		total_rmse = 0
		total_std = 0
		total_dice = 0
		total_iou = 0
		total_acc = 0
		total_kl = 0 
		for step, (inputs, targets, *args) in enumerate(train_data_iter): 
			if not multi_gpu:
				inputs = inputs.to(device)
				targets = targets.to(device)

			one_hot_targets = False
			if net_config["task"] == 'classification' and not data_config["multi_label"] or data_config["dataset"] == 'cityscapes':
				targets = utils.get_one_hot(targets, num_classes=net_config["output_size"], dataset=data_config["dataset"])
				one_hot_targets = True

			if net_config["network_type"] == 'deterministic':
				outputs = network.forward(inputs)
				loss = criterion(outputs, targets)

				if net_config["task"] == "regression": 
					total_rmse += float(torch.sqrt(loss))

			elif 'bayesian' in net_config["network_type"]:  
				outputs, kl = network.forward(inputs)
				total_kl += float(kl)

				if net_config["task"] == 'regression':
					rmse = []
					for i in range(outputs.shape[0]): 
						rmse.append(torch.sqrt(criterion(outputs[i], targets)))
					rmse = torch.stack(rmse, dim=0)
					likelihood = rmse.mean()
					total_rmse += float(likelihood)					
					std = rmse.std()
				else:
					likelihood = []
					for i in range(bayes_config["num_samples"]):
						likelihood.append(criterion(outputs[i], targets))
					likelihood = torch.mean(torch.stack(likelihood))
					std = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets, 
									 num_classes=net_config["output_size"], task=net_config["task"], 
									 multi_label=data_config["multi_label"], dataset=data_config["dataset"])
				
				total_std += float(std)
				loss = likelihood + kl_weight.get_weight(step, epoch)*kl
				
			if net_config["task"] == "classification": 
				#accuracy calculated as a mean of the batch
				total_acc += utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, data_config["multi_label"])
			
			elif net_config["task"] == "segmentation": 
				if train_config["loss_func"] == 'ce' or train_config["loss_func"] == 'focal':
					dim = 1 if net_config["network_type"] == 'deterministic' else 2
					outputs = torch.nn.Softmax(dim=dim)(outputs)

				if data_config["dataset"] == "cityscapes" or data_config["dataset"] == "isic":
					total_iou += float(utils.calculate_iou(outputs, targets, num_classes=net_config["output_size"], dataset=data_config["dataset"]))
				else:
					if train_config["loss_func"] == "dice":
						total_dice += float(1-loss)
					else:
						total_dice += float(utils.calculate_dice(input=outputs, target=targets, num_classes=net_config["output_size"]))

			total_loss += float(loss.detach())

			optimizer.zero_grad()
			if multi_gpu:
				fabric.backward(loss)
			else:
				loss.backward()
			optimizer.step()
			
		metrics["Train Time"] = time.time()-epoch_time
		metrics['Train Loss'] = total_loss/(max(step, 1))
		metrics["Train Accuracy"] = total_acc / max(step, 1)
		metrics["Train RMSE"] = total_rmse / max(step, 1)
		metrics["Train Std"] = total_std / max(step, 1)
		metrics["Train KL"] = total_kl / max(step, 1)
		metrics["Train Brier Score"] = total_std / max(step, 1)
		metrics["Train Dice"] = total_dice / max(step, 1)
		metrics["Train IoU"] = total_iou / max(step, 1)
		output_metrics_train = [metrics[key] for key in logging_keys if 'Train' in key and "Best" not in key] 
		output_metrics_val = []
		metric_formats = {
			"classification": {
				"print_format": "Fold {kfold}, Epoch {epoch}: Train Loss={Train Loss:.4f}, KL={Train KL:.4f}, Accuracy={Train Accuracy:.4f}, Brier Score={Train Brier Score:.4e}",
			},
			"regression": {
				"print_format": "Fold {kfold}, Epoch {epoch}: Train Loss={Train Loss:.4f}, Train KL={Train KL:.4f}, Train RMSE={Train RMSE:.4f}, Train Std={Train Std:.4f}",
			},
			"segmentation": {
				"print_format": "Fold {kfold}, Epoch {epoch}: Train Loss={Train Loss:.4f}, KL={Train KL:.4f}, Dice={Train Dice:.4f}, IoU={Train IoU:.4f}",
			}
		}

		if not multi_gpu or (multi_gpu and fabric.global_rank == 0):
			if not data_config["run_val"] and net_config["task"] == "segmentation":
				utils.save_pred(inputs, targets, outputs, epoch, output_path, data_config["dataset"], True, *args)
			
		print(metric_formats[net_config["task"]]["print_format"].format(**metrics, kfold=fold, epoch=epoch))

		# Evaluate the model on the Valset
		total_loss = 0
		total_rmse = 0
		total_std = 0
		total_acc = 0 #accuracy
		total_dice = 0
		total_iou = 0
		total_kl = 0
		if data_config["run_val"] and epoch%data_config["val_patience"]==0:
			val_data_iter = tqdm(val_loader, position=2)
			epoch_time = time.time()
			network.eval()
			with torch.no_grad():
				for step, (inputs, targets, *args) in enumerate(val_data_iter):
					if not multi_gpu:
						inputs = inputs.to(device)
						targets = targets.to(device)

					one_hot_targets = False
					if net_config["task"] == "classification" and not data_config["multi_label"] or data_config["dataset"] == 'cityscapes':
						targets = utils.get_one_hot(targets, num_classes=net_config["output_size"], dataset=data_config["dataset"])
						one_hot_targets = True

					if net_config["network_type"] == 'deterministic':
						outputs = network.forward(inputs)
						loss = criterion(outputs, targets)

						if net_config["task"] == "regression": 
							total_rmse += float(torch.sqrt(loss))

					elif 'bayesian' in net_config["network_type"]:  
						outputs, kl = network.forward(inputs)
						total_kl += float(kl)

						if net_config["task"] == 'regression':
							rmse = []
							for i in range(outputs.shape[0]): 
								rmse.append(torch.sqrt(criterion(outputs[i], targets)))
							rmse = torch.stack(rmse, dim=0)
							likelihood = rmse.mean()
							total_rmse += float(likelihood)					
							std = rmse.std()
						else:
							likelihood = []
							for i in range(bayes_config["num_samples"]):
								likelihood.append(criterion(outputs[i], targets))
							likelihood = torch.stack(likelihood)
							likelihood_std = torch.std(likelihood)
							likelihood = torch.mean(likelihood)
							std = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets,
									   num_classes=net_config["output_size"], task=net_config["task"], 
									   multi_label=data_config["multi_label"], dataset=data_config["dataset"])
						
						total_std += float(std)
						loss = likelihood + kl_weight.get_weight(step, epoch)*kl + likelihood_std
						
					if net_config["task"] == "classification": 
						#accuracy calculated as a mean of the batch
						total_acc += utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, data_config["multi_label"])
					elif net_config["task"] == "segmentation": 
						if train_config["loss_func"] == 'ce' or train_config["loss_func"] == 'focal':
							dim = 1 if net_config["network_type"] == 'deterministic' else 2
							outputs = torch.nn.Softmax(dim=dim)(outputs)

						if data_config["dataset"] == "cityscapes" or data_config["dataset"] == "isic":
							total_iou += float(utils.calculate_iou(outputs, targets, net_config["output_size"], dataset=data_config["dataset"]))
						else:
							if train_config["loss_func"] == "dice":
								total_dice += float(1-loss)
							else:
								total_dice += float(utils.calculate_dice(input=outputs, target=targets, num_classes=net_config["output_size"]))

					total_loss += float(loss)

		if data_config["run_val"] and epoch%data_config["val_patience"]==0:
			
			metrics["Val Time"] = time.time()-epoch_time
			metrics['Val Loss'] = total_loss/max(step, 1)	
			metrics["Val Accuracy"] = total_acc / max(step, 1)
			metrics["Val RMSE"] = total_rmse / max(step, 1)
			metrics["Val Std"] = total_std / max(step, 1)
			metrics["Val KL"] = total_kl / max(step, 1)
			metrics["Val Brier Score"] = total_std / max(step, 1)
			metrics["Val Dice"] = total_dice / max(step, 1)
			metrics["Val IoU"] = total_iou / max(step, 1)

			metric_formats = {
				"classification": {
					"print_format": "Fold {kfold}, Epoch {epoch}: Val Loss={Val Loss:.4f}, KL={Val KL:.4f}, Accuracy={Val Accuracy:.4f}, Brier Score={Val Brier Score:.4e}"
				},
				"regression": {
					"print_format": "Fold {kfold}, Epoch {epoch}: Val Loss={Val Loss:.4f}, Val KL={Val KL:.4f}, Val RMSE={Val RMSE:.4f}, Val Std={Val Std:.4f}"
				},
				"segmentation": {
					"print_format": "Fold {kfold}, Epoch {epoch}: Val Loss={Val Loss:.4f}, KL={Val KL:.4f}, Dice={Val Dice:.4f}, IoU={Val IoU:.4f}"
				}
			}

			print(metric_formats[net_config["task"]]["print_format"].format(**metrics, kfold=fold, epoch=epoch))

			if net_config["task"] == "segmentation" and not multi_gpu:
				utils.save_pred(inputs, targets, outputs, epoch, output_path, data_config["dataset"], False, *args)

			output_metrics_val = [metrics[key] for key in logging_keys if 'Val' in key and "Best" not in key] 

		if not multi_gpu or (multi_gpu and fabric.global_rank == 0):
			full_log.write([fold, epoch] + output_metrics_train + output_metrics_val)
			
			if multi_gpu:
				net_dict = network.module.state_dict()
			elif compiled:
				net_dict = network._orig_mod.state_dict()
			else:
				net_dict = network.state_dict()
			save_dict = {'epoch': epoch+1, 'network_state_dict': net_dict, 
				'current_train_time': time.time()-full_training_start_time,
				'optim_state_dict': optimizer.state_dict()}
				
			best_metrics = {
				"classification": {"train": ("Train Accuracy", "Best Train Accuracy", operator.gt), "val": ("Val Accuracy", "Best Val Accuracy", operator.gt)},
				"regression": {"train": ("Train RMSE", "Best Train RMSE", operator.lt), "val": ("Val RMSE", "Best Val RMSE", operator.lt)},
				"segmentation": {"train": ("Train Dice", "Best Train Dice", operator.gt), "val": ("Val Dice", "Best Val Dice", operator.gt)},
				"cityscapes": {"train": ("Train IoU", "Best Train IoU", operator.gt), "val": ("Val IoU", "Best Val IoU", operator.gt)},
				"isic": {"train": ("Train IoU", "Best Train IoU", operator.gt), "val": ("Val IoU", "Best Val IoU", operator.gt)}
			}

			if data_config["dataset"] == "cityscapes" or data_config["dataset"] == 'isic':
				metric_key, best_metric_key, comparison_op = best_metrics[data_config["dataset"]]["val"] if data_config["run_val"] else best_metrics[data_config["dataset"]]["train"]
			else: 
				metric_key, best_metric_key, comparison_op = best_metrics[net_config["task"]]["val"] if data_config["run_val"] else best_metrics[net_config["task"]]["train"]

		
			if comparison_op(metrics[metric_key], metrics[best_metric_key]):
				metrics[best_metric_key] = metrics[metric_key] 
				if data_config["run_val"]:
					if data_config["kfold"] > 1:
						torch.save(save_dict, output_path / f'checkpoint_best_val_fold{fold}.pth.tar')
					else:
						torch.save(save_dict, output_path / 'checkpoint_best_val.pth.tar')
				else:
					torch.save(save_dict, output_path / 'checkpoint_best_train.pth.tar')

			if data_config["kfold"] > 1:
				torch.save(save_dict, output_path / f'checkpoint_fold{fold}.pth.tar')
			else:
				torch.save(save_dict, output_path / 'checkpoint.pth.tar')

		if lr_scheduler is not None:
			if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
				lr_scheduler.step(metrics['Val Loss'])
				print(f"Learnign rate: {lr_scheduler._last_lr}")
			else:
				lr_scheduler.step()
				print(f"Learnign rate: {lr_scheduler.get_last_lr()}")

if data_config["kfold"] > 1: 
	f = data_config["kfold"]
	print(f"Running {f} fold Cross-validation")
	kf = KFold(n_splits=data_config["kfold"], shuffle=False)
	for fold, (train_index, val_index) in enumerate(kf.split(data_loader)):
		print(f"Running fold {fold+1}")
		# Split the data into train and Val sets
		train_dataset = torch.utils.data.Subset(data_loader, train_index)
		val_dataset = torch.utils.data.Subset(data_loader, val_index)

	   # Create a DataLoaders
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_config["batch_size"], shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

		if fold > 0 :
			#reset best metrics 
			metrics, logging_keys = utils.get_logging_keys(net_config["network_type"], net_config["task"], data_config["run_val"], dataset=data_config["dataset"]) 
			if net_config["network_type"] == "partial_bayesian":
				network = initialize_partial_bayes(fold)
			else: 
				network= model.get_model(net_config, bayes_config, pbayes_config)
			
			network.to(device)
			optimizer = utils.get_optimizer(network, opt_type=train_config["optimizer"], lr=train_config["lr"], weight_decay=train_config["weight_decay"])
			lr_scheduler = utils.get_lr_scheduler(config["Training"]["LR Scheduler"], optimizer)
		train_single_fold(network, train_loader, val_loader, optimizer=optimizer, full_log=full_log,
						  output_path=Path(config["Paths"]['save_path']), fold=fold, network_type=net_config["network_type"])
		
else: 
	print("Running a single split training loop")
	if multi_gpu:
		network, optimizers = fabric.setup(network, optimizer)
		data_loader, val_loader = fabric.setup_dataloaders(data_loader, val_loader)
	train_single_fold(network, data_loader, val_loader, optimizer=optimizer, full_log=full_log,
					output_path=Path(config["Paths"]['save_path']), fold=0, network_type=net_config["network_type"], multi_gpu=multi_gpu)
	
if not opts.no_date:
	log_run = utils.log_runs(config)