"""
Reparameterization layers:
source: https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/layers/variational_layers/

"""
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from torch.nn import Parameter
from itertools import repeat
import collections, math
import utils

def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False
        self.kl_loss = None

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()


class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparameterization, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result


        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)

        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            self.kl_loss = kl

        return out
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" \
            f"{self.in_features}, {self.out_features}, " \
            f"bias={self.bias}"


class Conv2dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super(Conv2dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        kernel_size = get_kernel_size(kernel_size, 2)

        self.mu_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv2d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)         

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            self.kl_loss = kl
            
        return out
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" \
            f"{self.in_channels}, {self.out_channels}, " \
            f"kernel_size={(self.kernel_size, self.kernel_size)}, " \
            f"padding={(self.padding, self.padding)}, " \
            f"stride={(self.stride, self.stride)}, " \
            f"bias={self.bias}" 


class Conv3dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 prior_mean,
                 prior_variance,
                 posterior_mu_init,
                 posterior_rho_init,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        """
        Implements Conv3d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super(Conv3dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        kernel_size = get_kernel_size(kernel_size, 3)
        self.mu_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.rho_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        tmp_result = sigma_weight * eps_kernel
        weight = self.mu_kernel + tmp_result

        if return_kl:
            kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.conv3d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)

        if return_kl:
            if self.bias:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight
            self.kl_loss = kl

        return out
            
    def __repr__(self):
        return f"{self.__class__.__name__}(" \
            f"{self.in_channels}, {self.out_channels}, " \
            f"kernel_size={(self.kernel_size, self.kernel_size)}, " \
            f"padding={(self.padding, self.padding)}, " \
            f"stride={(self.stride, self.stride)}, " \
            f"bias={self.bias}" 