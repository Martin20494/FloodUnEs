# Pytorch packages
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
from torchvision import datasets
from torchvision import transforms

# Spatial raster visualisation packages
import datashader as ds
from datashader.transfer_functions import shade
from datashader.transfer_functions import stack
from datashader.transfer_functions import dynspread
from datashader.transfer_functions import set_background
from datashader.colors import Elevation

import xrspatial
from xrspatial import proximity

import richdem as rd # for slope

from scipy import ndimage

# Raster manipulation packages
import rioxarray as rxr
from scipy.interpolate import griddata
import xarray as xr
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd

# Visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import seaborn as sns

# Data manipulation packages
import numpy as np
import pandas as pd

# Sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Other files
import random
import csv
import math
import pathlib

# Path
import os
import json
from pathlib import Path

# For variational approximator
from typing import Any, Optional
import torch.nn as nn
from torch import Tensor

from .dataCollection import dataCollection
from .rasterArray import blockshaped
from .bayesFuncClassification import BayesianNetwork, minibatch_weight
from .bayesFuncRegression import MLP_BBB
from .dataPrep import dataPreparation

torch.set_default_dtype(torch.float64)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class runBayesClassification():

    def __init__(self,
                 parameter_path_train,
                 setseed=2,
                 number_layers=9,
                 lr=1e-4,
                 batchsize=16*16,
                 num_workers=1,
                 parameter_path_test=None
                 ):

        # Set up set seed
        self.setseed = setseed

        # Set up parameters for getting data
        self.parameter_path_train = parameter_path_train
        # Call out file of paths to get general path
        # Train, val, and perhaps test
        if parameter_path_test != None:
            with open(self.parameter_path_train, "r") as para_path_train:
                para_path_train = json.load(para_path_train)
            Path(fr"{para_path_train['general_folder']}\model_classification_proportion").mkdir(parents=True,
                                                                                                exist_ok=True)
            self.general_folder = fr"{para_path_train['general_folder']}\model_classification_proportion"

            self.parameter_path_test = parameter_path_test
            with open(self.parameter_path_test, "r") as para_path_test:
                para_path_test = json.load(para_path_test)


            # Call data
            data_preparation = dataPreparation(parameter_path_train, False, para_path_test, setseed=self.setseed)

        else:
            with open(self.parameter_path_train, "r") as para_path_train:
                para_path_train = json.load(para_path_train)
            Path(fr"{para_path_train['general_folder']}\model_classification_proportion").mkdir(parents=True,
                                                                                                exist_ok=True)
            self.general_folder = fr"{para_path_train['general_folder']}\model_classification_proportion"

            # Call data
            data_preparation = dataPreparation(parameter_path_train, True, None, setseed=self.setseed)

        # Set data loaders
        trainloader, valloader, testloader, class_weight_new = data_preparation.pixel_dataloader_classification(
            batchsize=batchsize, num_workers=num_workers
        )
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.class_weight_new = class_weight_new

        # For test only if parameter_path_test = None
        # Call out trainloader, valoader, testloader

        # Set up parameters for your machine
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.device == 'cuda' else {}
        self.kwargs = kwargs


        # Set up layers and model
        self.model = BayesianNetwork(number_layers*1*1, 3, number_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def train_model(self, total_epochs):

        class1_acc_final = 0
        set_seed(self.setseed)

        for epoch in range(total_epochs):

            train_loss_total = 0.0

            self.model.train()
            for train_batchidx, (train_data, train_labels) in enumerate(self.trainloader):
                train_labels = train_labels.type(torch.LongTensor)
                train_data, train_labels = train_data.to(device), train_labels.to(device)

                self.optimizer.zero_grad()

                pi_weight = minibatch_weight(batch_idx=train_batchidx, num_batches=len(self.trainloader))

                train_loss = self.model.elbo(
                    inputs=train_data,
                    targets=train_labels,
                    alpha=self.class_weight_new.to(device),
                    gamma=2,
                    criterion=self.criterion,
                    n_samples=5,
                    w_complexity=pi_weight
                )

                train_loss_total += train_loss.item() * train_data.size(0)

                train_loss.backward()
                self.optimizer.step()


            correct = 0
            total = 0
            val_loss_total = 0

            prediction_labels = []
            validation_labels = []

            self.model.eval()
            with torch.no_grad():
                for val_batchidx, (val_data, val_labels) in enumerate(self.valloader):
                    val_labels = val_labels.type(torch.LongTensor)
                    val_data, val_labels = val_data.to(device), val_labels.to(device)

                    outputs = self.model(val_data)

                    pi_weight = minibatch_weight(batch_idx=val_batchidx, num_batches=len(self.valloader))

                    val_loss = self.model.elbo(
                        inputs=val_data,
                        targets=val_labels,
                        alpha=self.class_weight_new.to(device),
                        gamma=2,
                        criterion=self.criterion,
                        n_samples=5,
                        w_complexity=pi_weight
                    )

                    # Calculate total loss of VALIDATION
                    val_loss_total += val_loss.item() * val_data.size(0)

                    # Get predicted data to compare
                    probabilities = F.softmax(outputs)
                    _, predicted = torch.max(probabilities.data, 1)

                    # Get total correction
                    total += val_labels.size(0)
                    correct += torch.eq(predicted, val_labels).sum().item()

                    # Ref: https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch#:~:text=We%20can%20join%20two%20or,used%20to%20stack%20the%20tensors.
                    #    : https://discuss.pytorch.org/t/how-to-stack-over-for-loop/123214
                    #    : https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
                    # Accumulate labels into prediction_labels and validation_labels
                    prediction_labels.append(predicted)
                    validation_labels.append(val_labels)


            # Concatenate prediction_labels and validation_labels
            prediction_labels_concat = torch.cat(prediction_labels)
            prediction_labels_concat_np = prediction_labels_concat.cpu().detach().numpy()
            validation_labels_concat = torch.cat(validation_labels)
            validation_labels_concat_np = validation_labels_concat.cpu().detach().numpy()

            # Calculate confusion matrix
            cm = confusion_matrix(validation_labels_concat_np, prediction_labels_concat_np)

            # Normalise the diagonal entries
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # The diagonal entries are the accuracies of each class
            class0_acc, class1_acc, class2_acc = cm.diagonal() * 100

            # Overall accuracy
            accuracy = 100 * correct / total
            train_loss_total /= len(self.trainloader.dataset)
            val_loss_total /= len(self.valloader.dataset)

            # Save model
            if epoch >= 1500:
                torch.save(self.model.state_dict(), fr'{self.general_folder}\trained_model_{epoch}.pt')
                torch.save(self.model, fr'{self.general_folder}\full_model.pth')

            # Print results
            print('\nMAYBE accuracy increased: {:.2f} -> {:.2f}%\n'
                  ''.format(class1_acc_final, class1_acc))
            class1_acc_final = class1_acc

            print(f'Epoch: {epoch:04} |'
                  f'TrainLoss: {train_loss_total:.2f} |'
                  f'ValLoss: {val_loss_total:.2f} |'
                  f'AllAcc: {accuracy:.2f}% |'
                  f'NoAcc: {class0_acc:.2f}% |'
                  f'MaybeAcc: {class1_acc:.2f}% |'
                  f'YesAcc: {class2_acc:.2f}%\n')


    def retrain_model(self, pretrained_model_path, epoch_range):

        class1_acc_final = 0
        set_seed(self.setseed)

        # Call out pretrained model
        self.model.load_state_dict(torch.load(fr"{pretrained_model_path}"))

        for epoch in range(epoch_range[0], epoch_range[1], 1):

            train_loss_total = 0.0

            self.model.train()
            for train_batchidx, (train_data, train_labels) in enumerate(self.trainloader):
                train_labels = train_labels.type(torch.LongTensor)
                train_data, train_labels = train_data.to(device), train_labels.to(device)

                self.optimizer.zero_grad()

                pi_weight = minibatch_weight(batch_idx=train_batchidx, num_batches=len(self.trainloader))

                train_loss = self.model.elbo(
                    inputs=train_data,
                    targets=train_labels,
                    alpha=self.class_weight_new.to(device),
                    gamma=2,
                    criterion=self.criterion,
                    n_samples=5,
                    w_complexity=pi_weight
                )

                train_loss_total += train_loss.item() * train_data.size(0)

                train_loss.backward()
                self.optimizer.step()


            correct = 0
            total = 0
            val_loss_total = 0

            prediction_labels = []
            validation_labels = []

            self.model.eval()
            with torch.no_grad():
                for val_batchidx, (val_data, val_labels) in enumerate(self.valloader):
                    val_labels = val_labels.type(torch.LongTensor)
                    val_data, val_labels = val_data.to(device), val_labels.to(device)

                    outputs = self.model(val_data)

                    pi_weight = minibatch_weight(batch_idx=val_batchidx, num_batches=len(self.valloader))

                    val_loss = self.model.elbo(
                        inputs=val_data,
                        targets=val_labels,
                        alpha=self.class_weight_new.to(device),
                        gamma=2,
                        criterion=self.criterion,
                        n_samples=5,
                        w_complexity=pi_weight
                    )

                    # Calculate total loss of VALIDATION
                    val_loss_total += val_loss.item() * val_data.size(0)

                    # Get predicted data to compare
                    probabilities = F.softmax(outputs)
                    _, predicted = torch.max(probabilities.data, 1)

                    # Get total correction
                    total += val_labels.size(0)
                    correct += torch.eq(predicted, val_labels).sum().item()

                    # Ref: https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch#:~:text=We%20can%20join%20two%20or,used%20to%20stack%20the%20tensors.
                    #    : https://discuss.pytorch.org/t/how-to-stack-over-for-loop/123214
                    #    : https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
                    # Accumulate labels into prediction_labels and validation_labels
                    prediction_labels.append(predicted)
                    validation_labels.append(val_labels)


            # Concatenate prediction_labels and validation_labels
            prediction_labels_concat = torch.cat(prediction_labels)
            prediction_labels_concat_np = prediction_labels_concat.cpu().detach().numpy()
            validation_labels_concat = torch.cat(validation_labels)
            validation_labels_concat_np = validation_labels_concat.cpu().detach().numpy()

            # Calculate confusion matrix
            cm = confusion_matrix(validation_labels_concat_np, prediction_labels_concat_np)

            # Normalise the diagonal entries
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # The diagonal entries are the accuracies of each class
            class0_acc, class1_acc, class2_acc = cm.diagonal() * 100

            # Overall accuracy
            accuracy = 100 * correct / total
            train_loss_total /= len(self.trainloader.dataset)
            val_loss_total /= len(self.valloader.dataset)

            torch.save(self.model.state_dict(), fr'{self.general_folder}\trained_model_{epoch}.pt')
            torch.save(self.model, fr'{self.general_folder}\full_model.pth')

            # Print results
            print('\nMAYBE accuracy increased: {:.2f} -> {:.2f}%\n'
                  ''.format(class1_acc_final, class1_acc))
            class1_acc_final = class1_acc

            print(f'Epoch: {epoch:04} |'
                  f'TrainLoss: {train_loss_total:.2f} |'
                  f'ValLoss: {val_loss_total:.2f} |'
                  f'AllAcc: {accuracy:.2f}% |'
                  f'NoAcc: {class0_acc:.2f}% |'
                  f'MaybeAcc: {class1_acc:.2f}% |'
                  f'YesAcc: {class2_acc:.2f}%\n')


    def test_model(self, trained_model_path):

        predict_list = []
        test_list = []
        total = 0
        correct = 0

        self.model.load_state_dict(torch.load(fr"{trained_model_path}"))
        self.model.eval()

        for test_batchidx, (test_data, test_labels) in enumerate(self.testloader):
            test_labels = test_labels.type(torch.LongTensor)
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            outputs = self.model(test_data)

            probabilities = F.softmax(outputs)
            _, predicted = torch.max(probabilities.data, 1)

            total += test_labels.size(0)
            correct += torch.eq(predicted, test_labels).sum().item()

            test_list.append(test_labels)
            predict_list.append(predicted)


        # Pull data out and put them into numpy
        predict_list_np = [each_tensor.cpu().detach().numpy() for each_tensor in predict_list]
        predict_list_np_flatten = np.concatenate(predict_list_np).ravel()
        test_list_np = [each_tensor.cpu().detach().numpy() for each_tensor in test_list]
        test_list_np_flatten = np.concatenate(test_list_np).ravel()

        return predict_list_np_flatten, test_list_np_flatten





class runBayesRegressionProportion():

    def __init__(self,
                 parameter_path_train,
                 setseed=2,
                 number_layers=10,
                 lr=0.001,
                 batchsize=3200,
                 num_workers=1,
                 parameter_path_test=None,
                 resample=False
                 ):
        # Set up set seed
        self.setseed = setseed

        # Set up parameters for getting data
        self.parameter_path_train = parameter_path_train
        # Call out file of paths to get general path
        # Train, val, and perhaps test
        if parameter_path_test != None:
            with open(self.parameter_path_train, "r") as para_path_train:
                para_path_train = json.load(para_path_train)
            Path(fr"{para_path_train['general_folder']}\model_regression_proportion").mkdir(parents=True, exist_ok=True)
            self.general_folder = fr"{para_path_train['general_folder']}\model_regression_proportion"

            self.parameter_path_test = parameter_path_test
            with open(self.parameter_path_test, "r") as para_path_test:
                para_path_test = json.load(para_path_test)


            # Call data
            data_preparation = dataPreparation(parameter_path_train, False, para_path_test, setseed=self.setseed)

        else:
            with open(self.parameter_path_train, "r") as para_path_train:
                para_path_train = json.load(para_path_train)
            Path(fr"{para_path_train['general_folder']}\model_regression_proportion").mkdir(parents=True, exist_ok=True)
            self.general_folder = fr"{para_path_train['general_folder']}\model_regression_proportion"

            # Call data
            data_preparation = dataPreparation(parameter_path_train, True, None, setseed=self.setseed)

        # Set data loaders
        trainloader, valloader, testloader = data_preparation.pixel_dataloader_regression_proportion(
            batchsize=batchsize, num_workers=num_workers, resample=resample
        )
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # For test only if parameter_path_test = None
        # Call out trainloader, valoader, testloader

        # Set up parameters for your machine
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.device == 'cuda' else {}
        self.kwargs = kwargs


        # Set up layers and model
        self.model = MLP_BBB(number_layers, setseed=setseed).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_model(self, total_epochs):

        set_seed(self.setseed)

        min_val_loss = np.Inf

        for epoch in range(total_epochs):
            # Train
            train_loss_total = 0
            self.model.train()

            for train_batchidx, (train_xdata, train_ydata) in enumerate(self.trainloader):
                train_xdata = train_xdata.to(device)
                train_ydata = train_ydata.to(device)
                train_loss = self.model.sample_elbo(train_xdata, train_ydata, 1)
                train_loss_total += train_loss.item() * train_xdata.size(0)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # Validate
            val_loss_total = 0
            self.model.eval()

            with torch.no_grad():
                for val_batchidx, (val_xdata, val_ydata) in enumerate(self.valloader):
                    val_xdata = val_xdata.to(device)
                    val_ydata = val_ydata.to(device)

                    val_loss = self.model.sample_elbo(val_xdata, val_ydata, 1)

                    val_loss_total += val_loss.item() * val_ydata.size(0)

            train_loss_total /= len(self.trainloader.dataset)
            val_loss_total /= len(self.valloader.dataset)

            if val_loss_total < min_val_loss:
                print('\nValidation Loss Decreased: {:.6f} -> {:.6f}\n'
                      ''.format(min_val_loss, val_loss_total))

                min_val_loss = val_loss_total
                if epoch >= 100:
                    torch.save(self.model.state_dict(), fr'{self.general_folder}\trained_model_{epoch}.pt')
                    torch.save(self.model, fr'{self.general_folder}\full_model.pth')

            _results = [epoch, train_loss_total, val_loss_total]

            print(f'Epoch: {epoch:03} | '
                  f'Train Loss: {train_loss_total:.3f} |'
                  f'Validation Loss: {val_loss_total:.3f}')

    def retrain_model(self, pretrained_model_path, epoch_range):

        min_val_loss = np.Inf

        set_seed(self.setseed)

        # Call out pretrained model
        self.model.load_state_dict(torch.load(fr"{pretrained_model_path}"))

        for epoch in range(epoch_range[0], epoch_range[1], 1):
            # Train
            train_loss_total = 0
            self.model.train()

            for train_batchidx, (train_xdata, train_ydata) in enumerate(self.trainloader):
                train_xdata = train_xdata.to(device)
                train_ydata = train_ydata.to(device)
                train_loss = self.model.sample_elbo(train_xdata, train_ydata, 1)
                train_loss_total += train_loss.item() * train_xdata.size(0)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # Validate
            val_loss_total = 0
            self.model.eval()

            with torch.no_grad():
                for val_batchidx, (val_xdata, val_ydata) in enumerate(self.valloader):
                    val_xdata = val_xdata.to(device)
                    val_ydata = val_ydata.to(device)

                    val_loss = self.model.sample_elbo(val_xdata, val_ydata, 1)

                    val_loss_total += val_loss.item() * val_ydata.size(0)

            train_loss_total /= len(self.trainloader.dataset)
            val_loss_total /= len(self.valloader.dataset)

            if val_loss_total < min_val_loss:
                print('\nValidation Loss Decreased: {:.6f} -> {:.6f}\n'
                      ''.format(min_val_loss, val_loss_total))

                min_val_loss = val_loss_total
                torch.save(self.model.state_dict(), fr'{self.general_folder}\trained_model_{epoch}.pt')
                torch.save(self.model, fr'{self.general_folder}\full_model.pth')

            _results = [epoch, train_loss_total, val_loss_total]

            print(f'Epoch: {epoch:03} | '
                  f'Train Loss: {train_loss_total:.3f} |'
                  f'Validation Loss: {val_loss_total:.3f}')

    def test_model(self, trained_model_path):

        predict_list = []
        test_list = []
        sd_list = []

        self.model.load_state_dict(torch.load(fr"{trained_model_path}"))
        self.model.eval()

        for test_batchidx, (test_xdata, test_ydata) in enumerate(self.testloader):

            test_xdata = test_xdata.to(device)
            test_ydata = test_ydata.to(device)

            # Get outputs
            set_seed(self.setseed)
            outputs_lst = [self.model.forward(test_xdata).data.cpu().numpy().squeeze(1) for _ in range(100)]

            # Calculate mean
            output_T = np.array(outputs_lst).T
            output_mean = output_T.mean(axis=1)

            # Calculate std
            output_sd = output_T.std(axis=1)

            test_list.append(test_ydata)
            predict_list.append(output_mean)
            sd_list.append(output_sd)

        return predict_list, test_list, sd_list





class runBayesRegressionSD():

    def __init__(self,
                 parameter_path_train,
                 setseed=2,
                 number_layers=8,
                 lr=0.001,
                 batchsize=3200,
                 num_workers=1,
                 parameter_path_test=None,
                 resample=False
                 ):
        # Set up set seed
        self.setseed = setseed

        # Set up parameters for getting data
        self.parameter_path_train = parameter_path_train
        # Call out file of paths to get general path
        # Train, val, and perhaps test
        if parameter_path_test != None:
            with open(self.parameter_path_train, "r") as para_path_train:
                para_path_train = json.load(para_path_train)
            Path(fr"{para_path_train['general_folder']}\model_regression_SD").mkdir(parents=True, exist_ok=True)
            self.general_folder = fr"{para_path_train['general_folder']}\model_regression_SD"

            self.parameter_path_test = parameter_path_test
            with open(self.parameter_path_test, "r") as para_path_test:
                para_path_test = json.load(para_path_test)

            # Call data
            data_preparation = dataPreparation(parameter_path_train, False, para_path_test, setseed=self.setseed)

        else:
            with open(self.parameter_path_train, "r") as para_path_train:
                para_path_train = json.load(para_path_train)
            Path(fr"{para_path_train['general_folder']}\model_regression_SD").mkdir(parents=True, exist_ok=True)
            self.general_folder = fr"{para_path_train['general_folder']}\model_regression_SD"

            # Call data
            data_preparation = dataPreparation(parameter_path_train, True, None, setseed=self.setseed)

        # Set data loaders
        trainloader, valloader, testloader = data_preparation.pixel_dataloader_regression_sd(
            batchsize=batchsize, num_workers=num_workers, resample=resample
        )
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # For test only if parameter_path_test = None
        # Call out trainloader, valoader, testloader

        # Set up parameters for your machine
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.device == 'cuda' else {}
        self.kwargs = kwargs

        # Set up layers and model
        self.model = MLP_BBB(number_layers, setseed=setseed).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_model(self, total_epochs):

        set_seed(self.setseed)

        min_val_loss = np.Inf

        for epoch in range(total_epochs):
            # Train
            train_loss_total = 0
            self.model.train()

            for train_batchidx, (train_xdata, train_ydata) in enumerate(self.trainloader):
                train_xdata = train_xdata.to(device)
                train_ydata = train_ydata.to(device)
                train_loss = self.model.sample_elbo(train_xdata, train_ydata, 1)
                train_loss_total += train_loss.item() * train_xdata.size(0)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # Validate
            val_loss_total = 0
            self.model.eval()

            with torch.no_grad():
                for val_batchidx, (val_xdata, val_ydata) in enumerate(self.valloader):
                    val_xdata = val_xdata.to(device)
                    val_ydata = val_ydata.to(device)

                    val_loss = self.model.sample_elbo(val_xdata, val_ydata, 1)

                    val_loss_total += val_loss.item() * val_ydata.size(0)

            train_loss_total /= len(self.trainloader.dataset)
            val_loss_total /= len(self.valloader.dataset)

            if val_loss_total < min_val_loss:
                print('\nValidation Loss Decreased: {:.6f} -> {:.6f}\n'
                      ''.format(min_val_loss, val_loss_total))

                min_val_loss = val_loss_total
                if epoch >= 100:
                    torch.save(self.model.state_dict(), fr'{self.general_folder}\trained_model_{epoch}.pt')
                    torch.save(self.model, fr'{self.general_folder}\full_model.pth')

            _results = [epoch, train_loss_total, val_loss_total]

            print(f'Epoch: {epoch:03} | '
                  f'Train Loss: {train_loss_total:.3f} |'
                  f'Validation Loss: {val_loss_total:.3f}')

    def retrain_model(self, pretrained_model_path, epoch_range):

        min_val_loss = np.Inf

        set_seed(self.setseed)

        # Call out pretrained model
        self.model.load_state_dict(torch.load(fr"{pretrained_model_path}"))

        for epoch in range(epoch_range[0], epoch_range[1], 1):
            # Train
            train_loss_total = 0
            self.model.train()

            for train_batchidx, (train_xdata, train_ydata) in enumerate(self.trainloader):
                train_xdata = train_xdata.to(device)
                train_ydata = train_ydata.to(device)
                train_loss = self.model.sample_elbo(train_xdata, train_ydata, 1)
                train_loss_total += train_loss.item() * train_xdata.size(0)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # Validate
            val_loss_total = 0
            self.model.eval()

            with torch.no_grad():
                for val_batchidx, (val_xdata, val_ydata) in enumerate(self.valloader):
                    val_xdata = val_xdata.to(device)
                    val_ydata = val_ydata.to(device)

                    val_loss = self.model.sample_elbo(val_xdata, val_ydata, 1)

                    val_loss_total += val_loss.item() * val_ydata.size(0)

            train_loss_total /= len(self.trainloader.dataset)
            val_loss_total /= len(self.valloader.dataset)

            if val_loss_total < min_val_loss:
                print('\nValidation Loss Decreased: {:.6f} -> {:.6f}\n'
                      ''.format(min_val_loss, val_loss_total))

                min_val_loss = val_loss_total
                torch.save(self.model.state_dict(), fr'{self.general_folder}\trained_model_{epoch}.pt')
                torch.save(self.model, fr'{self.general_folder}\full_model.pth')

            _results = [epoch, train_loss_total, val_loss_total]

            print(f'Epoch: {epoch:03} | '
                  f'Train Loss: {train_loss_total:.3f} |'
                  f'Validation Loss: {val_loss_total:.3f}')

    def test_model(self, trained_model_path):

        predict_list = []
        test_list = []
        sd_list = []

        self.model.load_state_dict(torch.load(fr"{trained_model_path}"))
        self.model.eval()

        for test_batchidx, (test_xdata, test_ydata) in enumerate(self.testloader):
            test_xdata = test_xdata.to(device)
            test_ydata = test_ydata.to(device)

            # Get outputs
            set_seed(self.setseed)
            outputs_lst = [self.model.forward(test_xdata).data.cpu().numpy().squeeze(1) for _ in range(100)]

            # Calculate mean
            output_T = np.array(outputs_lst).T
            output_mean = output_T.mean(axis=1)

            # Calculate std
            output_sd = output_T.std(axis=1)

            test_list.append(test_ydata)
            predict_list.append(output_mean)
            sd_list.append(output_sd)

        return predict_list, test_list, sd_list