import argparse, random, time, os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd 
import dataset
import models.model as model
import utils
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.utils.data
"""======================================================================================="""
parse_in = argparse.ArgumentParser()
parse_in.add_argument('--run',      type=str, default=None, help='which model to run testing on')
parse_in.add_argument('--dataset',  type=str, default=None, help='which dataset to run testing on')
parse_in.add_argument('--corrupt',  type=str, default=None, help='corruption type and intensiry on CIFAR-10-C or ImageNet-C, separated with -')
parse_in.add_argument('--noise',    type=tuple, default=None, help='add noise to the dataset to check for robustness')
parse_in.add_argument('--save_pred', action='store_true', default=False, help='save tensor predictions for classification or segmentation task')

opts = parse_in.parse_args()
save_pred = True if opts.save_pred else False
run = opts.run
run_folder = Path(__file__).resolve().parent / 'output'/ run
config = utils.read_config(run_folder)
if config["Paths"]["dataset_path"] != '':
	dataset_path = config["Paths"]["dataset_path"]
else:
	dataset_path = os.path.join(Path(__file__).resolve().parent, 'datasets')

### LOAD CONFIGS
net_config = config["Network"]["Basic Setup"]
data_config = config["Training"]["Dataset"]
sparse_config = config["Network"]["Optional"]["Sparsity"]
train_config = config["Training"]["Parameters"]
bayes_config = config["Training"]["Bayesian Parameters"]
pbayes_config = config["Training"]["Partial Bayesian Parameters"]
lbayes_config = config["Training"]["Layer Bayesian Parameters"]
criterion = utils.get_loss_func(train_config["loss_func"], label_smoothing=0, dataset=data_config["dataset"]) #compute loss based on 0 label smoothing
nll_loss = True if train_config["loss_func"] == "nll_loss" else False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################################################
## LOGGING
metrics, logging_keys = utils.get_logging_keys(net_config["network_type"], net_config["task"], dataset=data_config["dataset"], test=True) 

if net_config["task"] == "regression" and data_config["dataset"].upper() == "TOYDATA":
	plotter = utils.ToyDataPlotter(num_samples=20)
#######################################################################################################
### LOAD DATA
data_config["dataset"] = opts.dataset if opts.dataset is not None else data_config["dataset"]
data_loader= dataset.get_dataloader(root=dataset_path, input_size=net_config["input_size"], data_config=data_config,
									test = True, noise=opts.noise, corruption=opts.corrupt)

#######################################################################################################
### NETWORK INITIALIZATION
network= model.get_model(net_config)
checkpoint = torch.load(f"{run_folder}/checkpoint_best_val.pth.tar", weights_only=False)
network.load_state_dict(checkpoint["network_state_dict"])
network.to(device)
# if net_config["network_type"] == 'partial_bayesian':
# 	checkpoint = torch.load(f"{run_folder}/init_checkpoint.pth.tar", weights_only=False)
# 	network = checkpoint["init_network"]
# elif net_config["network_type"] == "bayesian" and net_config["architecture"] != 'unet': 
# 	checkpoint = torch.load(f"{run_folder}/init_checkpoint.pth.tar", weights_only=False)
# 	network = checkpoint["init_model"]
# else:
# 	checkpoint = torch.load(f"{run_folder}/checkpoint.pth.tar", weights_only=False)
# 	network = checkpoint["init_network"]
# network.to(device)
log_file_name = "test" if opts.dataset is None else f"test_{opts.dataset}"
if opts.noise:
	log_file_name = log_file_name + f"noise_{opts.noise[0]}_{opts.noise[-1]}.csv" 
elif opts.corrupt:
	log_file_name = log_file_name + f"_{opts.corrupt}.csv"
else:
	log_file_name = log_file_name + '.csv'
full_log  = utils.CSVlogger(run_folder/log_file_name, logging_keys)
"""======================================================================================="""
def turn_off_gradients(model):
	for param in model.parameters():
		param.requires_grad_(False)

### START TESTING
def test_single_fold(network, test_loader, output_path, fold=0):
	test_data_iter = tqdm(test_loader, position=2)
	network.eval()
	with torch.no_grad():
		turn_off_gradients(network)
		confidences_, true_classes, predictions = [], [], []
		for step, (inputs, targets, *args) in enumerate(test_data_iter):
			inputs = inputs.to(device)
			targets = targets.to(device)
			one_hot_targets = False
			
			if net_config["task"] != 'segmentation' and not data_config["multi_label"]:
				metrics['Y'] = int(targets)
			if net_config["task"] == "classification" and not data_config["multi_label"]:
				targets = utils.get_one_hot(targets, num_classes=net_config["output_size"], dataset=data_config["dataset"])
				one_hot_targets = True
		
			inference_time = time.time()
			
			if net_config["network_type"] == 'deterministic':
				outputs = network(inputs)
				metrics['Test Time'] = time.time() - inference_time
				loss = criterion(outputs, targets)

				if net_config["task"] == 'regression':
					metrics['Y_pred'] = float(outputs)
					metrics['Test RMSE'] = float(torch.sqrt(loss))
				elif net_config["task"] == 'classification': 
					predicted = torch.argmax(outputs, -1)
					if nll_loss:
						#invert the log_ in log_softmax
						conf = torch.exp(outputs)
					else: 
						if data_config["multi_label"]:
							conf = F.sigmoid(outputs)
						else:
							conf = F.softmax(outputs, dim=-1) 

			elif 'bayesian' in net_config["network_type"]:  
				outputs, kl = network.forward(inputs)
				metrics['Test Time'] = time.time() - inference_time
	
				metrics['Test KL'] = float(kl)
				if net_config["task"] == 'regression':
					rmse = []
					for i in range(outputs.shape[0]): 
						rmse.append(torch.sqrt(criterion(outputs[i], targets)))
					rmse = torch.stack(rmse, dim=0)
					likelihood = rmse.mean()			
					std = rmse.std()
				else:
					likelihood = []
					for i in range(bayes_config["num_samples"]):
						likelihood.append(criterion(outputs[i], targets))
					likelihood = torch.mean(torch.stack(likelihood))
					std = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets, 
												num_classes=net_config["output_size"], task=net_config["task"],
												multi_label=data_config["multi_label"], dataset=data_config["dataset"])
				
				loss = likelihood
				std = float(std)
				
				if net_config["task"] == 'regression':
					metrics['Y_pred'] = float(outputs.mean())
					metrics['Test RMSE'] = float(likelihood.mean())
					metrics['Test Std'] = std
					
				elif net_config["task"] == 'classification': 
					if nll_loss:
						#invert the log_ in log_softmax
						conf = torch.exp(outputs)
					else: 
						if data_config["multi_label"]:
							conf = F.sigmoid(outputs)
						else:
							conf = F.softmax(outputs, dim=-1) 
					predicted = torch.mode(torch.argmax(conf, -1), dim=0).values
			metrics['Test NLL'] = float(loss)
			if net_config["task"] == "classification":    
				metrics['Test Accuracy'] = utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, data_config["multi_label"])
			
				if data_config["dataset"].upper() == "CIFAR100" or "IMAGENET" in data_config["dataset"].upper():
					metrics['Top5 Accuracy'] = utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, data_config["multi_label"], topk=5)
				metrics['Y_pred'] = int(predicted)
			
				if data_config["multi_label"]: 
						metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"] = utils.calculate_tp_tn_fp_fn(outputs, targets)

				if data_config["dataset"].upper() == "HALFMOON" or data_config["dataset"].upper() == "UNIFORM":
					metrics['X1'], metrics['X2'] =  inputs[0][0].item(),  inputs[0][1].item()
			
			elif net_config["task"] == "segmentation": 
				if train_config["loss_func"] == 'ce' or train_config["loss_func"] == 'focal':
					dim = 1 if net_config["network_type"] == 'deterministic' else 2
					outputs = torch.nn.Softmax(dim=dim)(outputs)
					conf = outputs.clone()
				else:
					true_class_indices = torch.argmax(outputs, dim=1)

			if "bayesian" in net_config["network_type"] and net_config["task"] != "regression": 
				metrics['Test Brier Score'] = std
				if net_config["task"] == 'classification':
					metrics['MaxSoftmax'] = float(conf.mean(0).max())
					metrics['VarSoftmax'] = float(conf.max(-1).values.var()) #variance of the maxsoftmax
					metrics['Y_conf'] = float(conf.mean(0)[0, metrics['Y_pred']]) ## confidence of the true class
			else: 
				if net_config["task"] == 'classification':
					metrics['MaxSoftmax'] = float(conf.mean(0).max())
					metrics['Y_conf'] = float(conf[0, metrics['Y_pred']])  ## confidence of the true class
					# confidences_.append(conf)
				if net_config["task"] != 'regression':
					metrics['Test Brier Score'] = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets, 
							num_classes=net_config["output_size"], task=net_config["task"], multi_label=data_config["multi_label"])
			if net_config["task"] == 'segmentation':
				if 'Test Dice' in metrics.keys():
					metrics['Test Dice'] = float(utils.calculate_dice(input=outputs, target=targets, num_classes=net_config["output_size"]))
				elif 'Test IoU' in metrics.keys():
					metrics['Test IoU'] = float(utils.calculate_iou(input=outputs, target=targets, num_classes=net_config["output_size"], dataset=data_config["dataset"]))
			metrics['Test Entropy'], metrics['Test Expected Entropy'] =utils.predictive_entropy(outputs, nll_loss=nll_loss, task=net_config["task"], multi_label=data_config["multi_label"])
					
			full_log.write([metrics[key] for key in logging_keys])

			if net_config["task"] == 'segmentation':
				pred_path = run_folder / f'test_preds_{fold}'
				if not pred_path.exists():
					pred_path.mkdir()
				if step % 100 == 0:
					utils.save_pred(inputs, targets, outputs, step, pred_path, data_config["dataset"], False, *args)
			
			elif net_config["task"] == 'regression' and data_config["dataset"].upper() == "TOYDATA":
				if len(plotter.sampled_points) < plotter.num_samples:
					plotter.sampled_points.append(float(inputs))
					plotter.model_outputs.append(float(outputs.mean()))
				if "bayesian" in net_config["network_type"]:
					plotter.pred_x.append(float(inputs))
					plotter.pred_mean.append(float(outputs.mean()))
					plotter.pred_std.append(float(outputs.std()))

			if save_pred:
				if step == 0:
					filename = f"{fold}"
					filename = filename + f"_{opts.dataset}" if opts.dataset is not None else filename
					filename = filename + f"_{opts.corrupt}" if opts.corrupt is not None else filename
				if net_config["task"] == 'segmentation':
					utils.append_to_hdf5(f"predictions_{filename}.h5", pred_path, step, inputs, targets, outputs, *args)
				elif net_config["task"] == 'classification':
					true_classes.append(targets)
					predictions.append(conf)

	if save_pred and net_config["task"] == 'classification':
		with open(run_folder/f'test_predictions_{filename}.pt', 'ab') as f:
			torch.save(torch.stack(predictions), f)
		with open(run_folder/f'test_targets_{filename}.pt', 'ab') as f:
			torch.save(torch.stack(true_classes), f)

if data_config["kfold"] > 1: 
	n_folds = data_config["kfold"]
	print(f"Running test for {n_folds} folds")
	kf = KFold(n_splits=data_config["kfold"], shuffle=False)
	for fold, (train_index, test_index) in enumerate(kf.split(data_loader)):
		test_dataset = torch.utils.data.Subset(data_loader, test_index)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
		net_fold, _ = utils.load_checkpoint(network, run_folder, cont_run=False, fold=fold)
		net_fold.to(device)
		print(f"Running test for fold {fold}")
		test_single_fold(net_fold, test_loader, run_folder, fold=fold)
else:
	net_fold, _ = utils.load_checkpoint(network, run_folder, cont_run=False)
	test_single_fold(network, data_loader, run_folder)

df = pd.read_csv(f"{str(run_folder)}/{log_file_name}")
means = df.mean() 
if net_config["task"] == 'classification':
	headings = 'Test Time\tTest Accuracy\tTest Entropy'
	values = f'{means["Test Time"]:0.4f}\t\t{means["Test Accuracy"]:.4f}\t\t{means["Test Entropy"]:.4f}'
elif net_config["task"] == 'segmentation':
	if 'Test Dice' in metrics.keys():
		headings = 'Test Time\tTest Dice\tTest Entropy'
		values = f'{means["Test Time"]:0.4f}\t\t{means["Test Dice"]:.4f}\t\t{means["Test Entropy"]:.4f}'
	elif 'Test IoU' in metrics.keys():
		headings = 'Test Time\tTest IoU\tTest Entropy'
		values = f'{means["Test Time"]:0.4f}\t\t{means["Test IoU"]:.4f}\t\t{means["Test Entropy"]:.4f}'
elif net_config["task"] == 'regression':
	headings = 'Test Time\tTest RMSE'
	values = f'{means["Test Time"]:0.4f}\t\t{means["Test RMSE"]:.4f}'
	if data_config["dataset"].upper() == "TOYDATA":
		plotter.plot_data(output_path=run_folder)

if "bayesian" in net_config["network_type"]:
	if net_config["task"]!="regression": 
		headings = headings + '\tTestBrierScore'
		values = values + f'\t\t{means["Test Brier Score"]:.4f}'
	else: 
		headings = headings + '\tTestStd'
		values = values + f'\t\t{means["Test Std"]:.4f}'
print(headings)
print(values)
print(f"Completed running test on run {run}")