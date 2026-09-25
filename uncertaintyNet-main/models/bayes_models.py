"""
Author: Zeinab Abboud
"""

import torch
import torch.nn as nn
import operator 

class VariationalNet(nn.Module):
	"""
    Wrapper for a neural network with Variational Bayesian layers.
    Accumulates KL divergence from specified reparameterized layers and returns multiple stochastic forward passes.
    """
	def __init__(self, model: nn.Module, reparam_layer_attrs: list=[], num_posterior_samples: int=0):
		"""
		Args:
			model (nn.Module): The base neural network model.
			reparam_layer_attrs (list): List of attribute names for reparameterized layers.
			num_posterior_samples (int): Number of posterior samples to draw in forward pass.
		"""
		super().__init__()
		self.base_model = model
		self.reparam_layer_attrs = reparam_layer_attrs
		self.num_posterior_samples = num_posterior_samples
		self.kl_div = None

	def forward(self, x):
		"""
		Forward pass with multiple stochastic samples.
		Args:
			x (Tensor): Input tensor.
		Returns:
			Tuple[Tensor, float]: Stacked outputs and accumulated KL divergence.
		"""
		samples = []
		for _ in range(self.num_posterior_samples):
			samples.append(self.base_model(x))
		out = torch.stack(samples, dim=0)
		self.kl_div = 0
		for reparam_layer in self.reparam_layer_attrs:
			kl = operator.attrgetter(reparam_layer)(self.base_model).kl_loss
			self.kl_div += kl
		return out, self.kl_div
	

class VariationalLastLayerNet(VariationalNet):
	"""
    Wrapper for a neural network where only the last layer is Variational Bayesian.
    Applies Bayesian sampling to the last layer and accumulates its KL divergence.
    """
	def __init__(self, reparam_last_layer:nn.Module=None, *args, **kwargs):
		"""
		Args:
			reparam_last_layer (nn.Module): The last Bayesian layer.
		"""
		super().__init__(*args, **kwargs)
		self.last_layer = reparam_last_layer

	def forward(self, x):
		"""
		Forward pass with multiple stochastic samples for the last layer.
		Args:
			x (Tensor): Input tensor.
		Returns:
			Tuple[Tensor, float]: Stacked outputs and KL divergence from the last layer.
		"""
		out = self.base_model(x)
		samples = []
		for _ in range(self.num_posterior_samples):
			samples.append(self.last_layer(out))
		out = torch.stack(samples, dim=0)

		self.kl_div = self.last_layer.kl_loss

		return out, self.kl_div
