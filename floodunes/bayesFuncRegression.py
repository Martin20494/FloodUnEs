import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.distributions import Normal
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

from torch.utils.data import WeightedRandomSampler

from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

import datashader as ds
from datashader.transfer_functions import shade
from datashader.transfer_functions import stack
from datashader.transfer_functions import dynspread
from datashader.transfer_functions import set_background
from datashader.colors import Elevation

import xrspatial
from xrspatial import proximity

from scipy import ndimage

from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error

import numpy as np
import math
import random

import rioxarray as rxr
import xarray as xr

import richdem as rd # for slope

import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

import matplotlib.pyplot as plt

# Ref: https://github.com/pytorch/pytorch/issues/2155
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from typing import Any, Optional

import resreg


# Ref: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"Random seed set as {seed}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv_BBB(nn.Module):
    """
        Layer of our Conv BNN
    """

    def __init__(self, input_features: int,
                 output_features: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0,
                 dilation: Optional[int] = 1,
                 prior_var: Optional[float] = 1.,
                 setseed: Optional[int] = 2) -> None:
        """
            Initialization of our layer: Our prior is a normal distribution
            centered in 0 and of variance 20/
        """
        # initialize layers
        super().__init__()

        # Seed
        set_seed(setseed)

        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # initialize mu and rho parmaeters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features, *self.kernel_size))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features, *self.kernel_size))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, x):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(device)
        self.w = self.w_mu.to(device) + torch.log(1 + torch.exp(self.w_rho.to(device))) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(device)
        self.b = self.b_mu.to(device) + torch.log(1 + torch.exp(self.b_rho.to(device))) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data.to(device), torch.log(1 + torch.exp(self.w_rho.to(device))))
        self.b_post = Normal(self.b_mu.data.to(device), torch.log(1 + torch.exp(self.b_rho.to(device))))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)


# Ref: https://stackoverflow.com/questions/75693007/pytorch-runtimeerror-mat1-mat2-shapes-cannot-be-multiplied

class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=1., setseed=2):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()

        # Get seed
        set_seed(setseed)

        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, x):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(device)
        self.w = self.w_mu.to(device) + torch.log(1 + torch.exp(self.w_rho.to(device))) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(device)
        self.b = self.b_mu.to(device) + torch.log(1 + torch.exp(self.b_rho.to(device))) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data.to(device), torch.log(1 + torch.exp(self.w_rho.to(device))))
        self.b_post = Normal(self.b_mu.data.to(device), torch.log(1 + torch.exp(self.b_rho.to(device))))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(x, self.w, self.b)


class MLP_BBB(nn.Module):

    def __init__(self, input_features, noise_tol=.1, prior_var=1., setseed=2):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()

        # Get number of layers
        self.number_layers = input_features

        # Conv
        self.conv1 = Conv_BBB(input_features, 6, 1, stride=1, padding=1, setseed=setseed)
        self.conv2 = Conv_BBB(6, 8, 1, stride=1, padding=0, setseed=setseed)
        self.conv3 = Conv_BBB(8, input_features, 1, stride=1, padding=0, setseed=setseed)

        # Linear
        self.hidden1 = Linear_BBB(input_features, 120, prior_var=prior_var, setseed=setseed)
        self.hidden2 = Linear_BBB(120, 84, prior_var=prior_var, setseed=setseed)
        self.out = Linear_BBB(84, 1, prior_var=prior_var, setseed=setseed)
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 1)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 1)

        x = x.view(-1, self.number_layers)

        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden1.log_prior + self.hidden2.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden1.log_post + self.hidden2.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        outputs = torch.zeros(samples, target.shape[0]).to(device)  # 이거를 to(device 안했더니 에러 났음. 왜?)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss

