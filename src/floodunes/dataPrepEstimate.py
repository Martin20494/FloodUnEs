# Pytorch packages
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

# Visualisation packages
import matplotlib.pyplot as plt

# Data manipulation packages
import numpy as np

# Sklearn packages
from sklearn.model_selection import train_test_split

# Random package
import random

# Oversampling package
import resreg

# Path
import os

# Other packages
from .dataCollection import dataCollection
from .rasterArray import blockshaped


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

class pixelDatasetGenerationEstimate():

    def __init__(self, x_flatten_channel, setseed=2):
        self.x_flatten_channel = x_flatten_channel
        self.setseed = setseed

    def pixel_dataset_generation_estimate(self):

        # FOR X ==================================
        # Reshape big arrays into small arrays - x
        reshape_into_small_batches_list_x = []
        small_array_size = 1

        for i, big_array_x in enumerate(self.x_flatten_channel):  # looping through each channel. Each channel here was flattened already
            # First, np.split is used to split each flattened channel into number-of-channels chunks/small arrays.
            # The result is a list of number-of-channels chunks/small arrays.
            # (Ex: No. channels is 4, split the flatten array into 4, the length of flattened array is 1080096)

            # Second, np.stack is used to stack all splitted small arrays into a nxdimension array.
            # (Ex: No. of small arrays is 4, after using np.stack, the new array shape is (4, 45024))

            # Third, blockshaped is used to create an array with first dimension is for number of batches, second and third dimension are the size for each batch.
            # (Ex: as we want to create batch size 1x1 so we will have 4*45024 = 180096 batches for each channel)

            # Forth, append all blockshaped array into a list. There will be four blockshapeds represent for the number of channels
            # (Ex: Here we have 4 channels so there will be 4 elements in the list)
            small_array_xtrain = blockshaped(
                np.stack(np.split(big_array_x, big_array_x.shape[0] / small_array_size)), 1, small_array_size)
            reshape_into_small_batches_list_x.append(small_array_xtrain)

        # Stack all elements in the list into an array
        # (Ex: here the array shape will be (4, 180096, 1, 1))
        reshape_into_small_batches_array_x = np.stack(reshape_into_small_batches_list_x)

        # -------------------------------------
        # Reshape to put the batches in front of the channels - x
        switch_batches_channels_list_x = []

        for i in range(reshape_into_small_batches_array_x.shape[
                           1]):  # loop through the length of each channel. Here is 180096
            # Pull all cell values at the same indices through out all channels
            # (Ex: value of 1st cell in channel 1 is -27.1583179, channel 2 is -27.11727658, channel 3 is 0, channel 4 is 0.01680811)
            one_batch_all_channels_x = np.vstack(
                [
                    reshape_into_small_batches_array_x[n][i].reshape(-1, small_array_size) for n in
                    range(reshape_into_small_batches_array_x.shape[0])
                ]
            )
            # Reshape into (4, 1, 1) instead of (4, 1). In that, 1st dimension is number of channels, second and third are sizes of a batch
            # Ref: https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array
            one_batch_all_channels_x = one_batch_all_channels_x[:, :, None]
            switch_batches_channels_list_x.append(one_batch_all_channels_x)

        switch_batches_channels_array_x = np.stack(switch_batches_channels_list_x)
        switch_batches_channels_array_x = switch_batches_channels_array_x.astype('float64')

        return switch_batches_channels_array_x


class PixelEstimate():

    def __init__(self, x):
        # data loading
        self.x = torch.Tensor(x)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.n_samples


class dataPreparationEstimate:
    def __init__(self, parameter_path, setseed=2):

        self.para_path = parameter_path
        self.setseed = setseed


    def estimate_data(self, type):

        if type == 'classification_proportion':

            # Get estimate collection
            estimate_collection = dataCollection(
                self.para_path['estimate']['general_folder'],
                self.para_path['estimate']['geometry_domain_list'],
                self.para_path['estimate']['path_elev'],
                self.para_path['estimate']['path_wd'],
                self.para_path['estimate']['path_wse'],
                self.para_path['estimate']['path_proportion'],
                self.para_path['estimate']['path_manning'],
                self.para_path['estimate']['path_roughness']
            )

        elif type == 'regression_proportion':

            # Get estimate collection
            estimate_collection = dataCollection(
                self.para_path['estimate']['general_folder'],
                self.para_path['estimate']['geometry_domain_list'],
                self.para_path['estimate']['path_elev'],
                self.para_path['estimate']['path_wd'],
                self.para_path['estimate']['path_wse'],
                self.para_path['estimate']['path_proportion'],
                self.para_path['estimate']['path_manning'],
                self.para_path['estimate']['path_roughness']
            )

        else: # for regression_sd

            # Get estimate collection
            estimate_collection = dataCollection(
                self.para_path['estimate']['general_folder'],
                self.para_path['estimate']['geometry_domain_list'],
                self.para_path['estimate']['path_elev'],
                self.para_path['estimate']['path_wd'],
                self.para_path['estimate']['path_wse'],
                self.para_path['estimate']['path_proportion'],
                self.para_path['estimate']['path_manning'],
                self.para_path['estimate']['path_roughness']
            )


        # Get dataframe
        estimate_df = estimate_collection.loadpara_into_dataframe_estimate(
            type=type,
            name_csv='estimate'
        )

        # Write out for checking
        estimate_df.to_csv(fr"{self.para_path['estimate']['general_folder']}/estimate_df.csv", index=False)

        # Flatten out values
        x_estimate_flatten_channel = estimate_df.loc[:, list(estimate_df.columns[2:])].to_numpy().T

        # Get pixels values across simulations
        estimate_pixel_dataset = pixelDatasetGenerationEstimate(
            x_estimate_flatten_channel,
            setseed=self.setseed
        )

        return estimate_pixel_dataset


    def pixel_dataloader_estimate(self,
                                  type,
                                  batchsize=2048,
                                  num_workers=1):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == 'cuda' else {}

        # Call estimate pixel dataset
        estimate_pixel_dataset = self.estimate_data(type)

        set_seed(self.setseed)
        # FOR ESTIMATE
        estimateloader = DataLoader(
            dataset=PixelEstimate(
                estimate_pixel_dataset.pixel_dataset_generation_estimate()
            ),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        return estimateloader







