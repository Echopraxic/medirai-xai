"""
Author: Zeinab Abboud
"""

import os
import operator
import torch
import torch.nn as nn
import models.bayes_layers as bayeslayers
import models.bayes_models as bayesmodels

"""==================================================="""
"""================= SETUP Utilities ================="""
"""==================================================="""

def get_nested_attr(model, attr_path):
	"""
	source: discuss.pytorch.org/t/how-to-replace-a-layer-or-module-in-a-pretrained-network/60068/3
	Retrieve a nested attribute and its parent in a PyTorch model using a dot-separated string path.

	Args:
		model (nn.Module): The PyTorch model to navigate.
		attr_path (str): Dot-separated path to the nested attribute (e.g., "layer1.conv1").

	Returns:
		tuple: (parent module, nested attribute path, final attribute name, layer/module at that path)
	"""
	if '.weight' in attr_path or '.bias' in attr_path:
		attr_path = attr_path.split('.weight')[0]
		attr_path = attr_path.split('.bias')[0]

	attrs = attr_path.split('.')

	# Traverse up to the second-to-last attribute
	parent_attr = model
	for attr in attrs[:-1]:
		parent_attr = getattr(parent_attr, attr)
	
	# Get the final layer
	layer = getattr(parent_attr, attrs[-1])
	
	# Return the parent attribute, nested_attribute name, final attribute name, and the layer
	return parent_attr, attr_path, attrs[-1], layer


def build_variational_model(
	model: nn.Module,
	net_type: str,
	output_path: str,
	bayes_config: dict,
	pbayes_config: dict,
	lbayes_config: dict,
	multi_gpu: bool=False,
	fabric=None,
):
	"""
	Replace selected layers in a model with Bayesian layers and save checkpoint.

	Args:
		model (nn.Module): The PyTorch model to modify.
		net_type (str): Type of the network (e.g., "bayesian", "partial_bayesian").
		output_path (str): Path to save the checkpoint.
		bayes_config (dict): Bayesian configuration.
		pbayes_config (dict): Partial Bayesian configuration.
		lbayes_config (dict): Layer-wise Bayesian configuration.

	Returns:
		nn.Module: Model with selected layers replaced by Bayesian layers.
	"""
	# Determine which layers to replace
	state_keys = list(model.state_dict().keys())

	if net_type == 'partial_bayesian':
		if lbayes_config["first_and_last"]:
			layer_selection = "first_and_last"
			layer_indices = [0, -1]
		else: 
			if lbayes_config["last_layer"]:
				layer_selection = "last"
				layer_indices = [-1]
			else:
				layer_selection = "first"
				layer_indices = [0]
	elif net_type == 'bayesian':
		layer_selection = "all"
		layer_indices = [i for i, l in enumerate(state_keys) if 'weight' in l and 'bn' not in l]
	else:
		raise ValueError(f"Unknown network type: {net_type}")

	custom_layer_attr = []
	for index in layer_indices:
		attr_name = state_keys[index]
		param_type = '.bias' if 'bias' in attr_name else '.weight'
		attr_name = attr_name.split(param_type)[0]
		parent_attr, nested_attr_names, child_attr_name, layer = get_nested_attr(model, attr_name)
		has_bias = True if hasattr(layer, 'bias') and layer.bias is not None else False

		if 'Linear' in layer.__class__.__name__:
			repl_layer = bayeslayers.LinearReparameterization(
				in_features=layer.in_features, out_features=layer.out_features, bias=has_bias,
				prior_mean=bayes_config['prior_mu'], prior_variance=bayes_config['prior_variance'],
				posterior_mu_init=bayes_config['posterior_mu'], posterior_rho_init=bayes_config['posterior_rho'])
		elif 'Conv2d' in layer.__class__.__name__:
			repl_layer = bayeslayers.Conv2dReparameterization(
				in_channels=layer.in_channels, out_channels=layer.out_channels,
				kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
				dilation=layer.dilation, groups=layer.groups, bias=has_bias,
				prior_mean=bayes_config['prior_mu'], prior_variance=bayes_config['prior_variance'],
				posterior_mu_init=bayes_config['posterior_mu'], posterior_rho_init=bayes_config['posterior_rho'])
		elif 'Conv3d' in layer.__class__.__name__:
			repl_layer = bayeslayers.Conv3dReparameterization(
				in_channels=layer.in_channels, out_channels=layer.out_channels,
				kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
				dilation=layer.dilation, groups=layer.groups, bias=has_bias,
				prior_mean=bayes_config['prior_mu'], prior_variance=bayes_config['prior_variance'],
				posterior_mu_init=bayes_config['posterior_mu'], posterior_rho_init=bayes_config['posterior_rho'])
		else:
			raise ValueError(f"Layer {layer.__class__.__name__} not supported")

		setattr(parent_attr, child_attr_name, repl_layer)
		custom_layer_attr.append(nested_attr_names)

	# Select wrapper class
	if layer_selection == "last":
		setattr(parent_attr, child_attr_name, Identity())
		variational_model = bayesmodels.VariationalLastLayerNet(model=model, reparam_last_layer=repl_layer, num_posterior_samples=bayes_config['num_samples'])
	else:
		variational_model = bayesmodels.VariationalNet(model=model, reparam_layer_attrs=custom_layer_attr, num_posterior_samples=bayes_config['num_samples'])

	custom_checkpoint = {
		'init_state_dict': variational_model.state_dict()
		# 'init_network': variational_model
	}

	if (multi_gpu and fabric.global_rank == 0) or not multi_gpu:
		if not os.path.isdir(output_path):
			os.makedirs(output_path)
		torch.save(custom_checkpoint, output_path + f'/init_checkpoint.pth.tar')

	return variational_model


class Identity(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x


"""==================================================="""
"""================== LOSS Utilities ================="""
"""==================================================="""
class KL_Weight():
	"""_summary_
	 REF: Weight Uncertainty in Neural Networks by Blundell (2015)
	 the weight of the KL term is set to 2^(M-i) / 2^M - 1
	 where i is the minibatch index, 
	 M is the total number of minibatches (length of the dataloader)
	"""
	def __init__(self, model: nn.Module,
			  		method: str, 
		  			m_train_batch: int, 
					m_val_batch: int,
					len_train_data: int,
					len_val_data: int,
					num_epochs: int,
					kl_weight_lim: list,
					hi2low=False):
		super().__init__()

		self.model = model
		self.method = method
		self.init_weight = kl_weight_lim[0]
		self.targetweight = kl_weight_lim[1]
		self.num_epochs = num_epochs
		#controls whether to increase or decrease the weight a function of epochs
		# if hi2low (high to low) is set to true, the KL weight would be highest 
		# initially and decreasing with number of epochs -- implemented for method="epoch"
		self.h2l = hi2low 

		self.len_train_data = len_train_data
		self.len_val_data = len_val_data

		self.train_batch = m_train_batch
		self.val_batch = m_val_batch

		if method == "bayes_params":
			self.layer_weight = self.relative_to_bayes_params()
	
	def relative_to_bayes_params(self):
		num_bayes_params = 0
		if hasattr(self.model, 'last_layer'):
			if "Linear" in self.model.last_layer.__class__.__name__:
				num_bayes_params = self.model.last_layer.mu_weight.numel()
			else: 
				num_bayes_params = self.model.last_layer.mu_kernel.numel()
				
			if self.model.last_layer.bias:
				num_bayes_params += self.model.last_layer.mu_bias.numel()
		else:
			for l in self.model.reparam_layer_attrs:
				num_bayes_params += sum(p.numel() for name, p in operator.attrgetter(l)(self.model.base_model).named_parameters() if 'mu' in name)
		total_num_params = sum(p.numel() for name, p in self.model.base_model.named_parameters() if 'bn' not in name and 'rho' not in name and 'mask' not in name)
		kl_weight = num_bayes_params / total_num_params
		print("KL Weight relative to Num Bayes Params is ", kl_weight)
		return kl_weight

	def get_weight(self, step:int, epoch: int, training=True):
		if self.method == "blundell": 
			if training:
				return 2**(self.train_batch-step) / (2**self.train_batch -1)
			else:
				return 2**(self.val_batch-step) / (2**self.val_batch -1)
			
		elif self.method == "num_samples": 
			if training: 
				return 1/self.len_train_data
			else: 
				return 1/self.len_val_data
			
		elif self.method == "epoch": 
			if self.h2l:
				return self.targetweight - (self.targetweight - self.init_weight) * (epoch / self.num_epochs)
			else:
				return self.init_weight + (self.targetweight - self.init_weight) * (epoch / self.num_epochs)
			
		elif self.method == "batch_size":
			return 1/self.train_batch
		
		elif self.method == "bayes_params":
			return self.layer_weight
		
		elif isinstance(self.method, (float, int)):
			return self.method
