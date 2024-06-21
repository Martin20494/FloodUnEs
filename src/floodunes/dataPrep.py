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

class pixelDatasetGeneration():

    def __init__(self, x_flatten_channel, y_flatten_channel, setseed=2):
        self.x_flatten_channel = x_flatten_channel
        self.y_flatten_channel = y_flatten_channel
        self.setseed = setseed

    def pixel_dataset_generation(self):

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

        # FOR Y ===================================
        # Reshape big arrays into small arrays - y
        reshape_into_small_batches_list_y = []

        small_array_y = blockshaped(
            np.stack(np.split(self.y_flatten_channel, self.y_flatten_channel.shape[0] / small_array_size)), 1,
            small_array_size)
        reshape_into_small_batches_list_y.append(small_array_y)

        reshape_into_small_batches_array_y = np.stack(reshape_into_small_batches_list_y)

        # -----------------------------------
        # Reshape to put the batches in front of the channels - y train
        switch_batches_channels_list_y = []

        for i in range(reshape_into_small_batches_array_y.shape[1]):
            one_batch_all_channels_y = np.vstack(
                [
                    reshape_into_small_batches_array_y[n][i].reshape(-1, small_array_size) for n in
                    range(reshape_into_small_batches_array_y.shape[0])
                ]
            )
            switch_batches_channels_list_y.append(one_batch_all_channels_y)

        switch_batches_channels_array_y = np.stack(switch_batches_channels_list_y)

        # ---------------------------------
        # Reshape one more time - y train (reduce dimension)
        switch_batches_channels_array_y = switch_batches_channels_array_y.flatten()
        switch_batches_channels_array_y = switch_batches_channels_array_y.astype('float64')

        # FOR DATASET ============================
        datasets = [
            switch_batches_channels_array_x,
            switch_batches_channels_array_y
        ]

        return datasets


    def weight_and_sampler(self, coef):

        # FOR WEIGHT -------------------
        # Ref: https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/
        # Class analysis ---------------
        # Get y_train
        if coef is None:
            coef = [1.2, 5, 1.7]
        y_train = self.pixel_dataset_generation()[1].astype('int64')

        # Count classes
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)

        # Get weights
        class_weights = []
        for count in class_counts:
            weight = 1 / (count / total_samples)
            class_weights.append(weight)

        # Convert to Tensor
        class_weights = torch.Tensor(class_weights)

        # Class weight new
        class_weights_new = torch.Tensor([
            class_weights[0]*coef[0],
            class_weights[1]*coef[1],
            class_weights[2]*coef[2]
        ])


        # FOR SAMPLER -------------------
        # Ref: https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
        #    : https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
        #    : https://discuss.pytorch.org/t/weighted-random-sampler-still-unbalanced/109702/6
        # Get weights
        initial_weight = 1 / class_counts
        samples_weight = np.array([initial_weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)

        set_seed(self.setseed)

        # Weighted Random Sampler
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)

        return class_weights_new, sampler




class Pixel():

    def __init__(self, xy, transform=None):
        # data loading
        self.x = torch.Tensor(xy[0])
        self.y = torch.Tensor(xy[1])
        self.n_samples = xy[0].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples





class dataPreparation:
    def __init__(self, parameter_path, split=True, setseed=2):

        self.para_path = parameter_path
        self.split = split
        self.setseed = setseed


    def split_data_classification(self):

        # Get train collection
        train_collection = dataCollection(
            self.para_path['train']['general_folder'],
            self.para_path['train']['geometry_domain_list'],
            self.para_path['train']['path_elev'],
            self.para_path['train']['path_wd'],
            self.para_path['train']['path_wse'],
            self.para_path['train']['path_proportion'],
            self.para_path['train']['path_manning'],
            self.para_path['train']['path_roughness'],
            self.para_path['train']['path_sd']
        )

        # Get dataframe for machine learning
        ml_train_df = train_collection.loadpara_into_dataframe_classification()

        if self.split == True:

            # Split data into train, validation, and test IF SPLIT = TRUE
            train_val_df, test_df = train_test_split(ml_train_df, random_state=42, train_size=0.8, shuffle=True)
            train_df, val_df = train_test_split(train_val_df, random_state=42, train_size=0.75, shuffle=True)

            # Write out just incase
            train_df.to_csv(fr"{self.para_path['train']['general_folder']}\train_df_classification.csv", index=False)
            val_df.to_csv(fr"{self.para_path['train']['general_folder']}\val_df_classification.csv", index=False)
            test_df.to_csv(fr"{self.para_path['train']['general_folder']}\test_df_classification.csv", index=False)

            # Flatten data
            x_train_flatten_channel = train_df.loc[:, list(ml_train_df.columns[2:11])].to_numpy().T
            y_train_flatten_channel = train_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_val_flatten_channel = val_df.loc[:, list(ml_train_df.columns[2:11])].to_numpy().T
            y_val_flatten_channel = val_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_test_flatten_channel = test_df.loc[:, list(ml_train_df.columns[2:11])].to_numpy().T
            y_test_flatten_channel = test_df.loc[:, ml_train_df.columns[-1]].to_numpy().T


        else:
            # Split data into train and validation only IF SPLIT = FALSE
            train_df, val_df = train_test_split(ml_train_df, random_state=42, train_size=0.8, shuffle=True)

            # Get test collection
            test_collection = dataCollection(
                self.para_path['test']['general_folder'],
                self.para_path['test']['geometry_domain_list'],
                self.para_path['test']['path_elev'],
                self.para_path['test']['path_wd'],
                self.para_path['test']['path_wse'],
                self.para_path['test']['path_proportion'],
                self.para_path['test']['path_manning'],
                self.para_path['test']['path_roughness'],
                self.para_path['test']['path_sd']
            )

            # Get dataframe for machine learning
            test_df = test_collection.loadpara_into_dataframe_classification(name_csv='test')

            # Write out just incase
            train_df.to_csv(fr"{self.para_path['train']['general_folder']}\train_df_classification.csv", index=False)
            val_df.to_csv(fr"{self.para_path['train']['general_folder']}\val_df_classification.csv", index=False)
            test_df.to_csv(fr"{self.para_path['test']['general_folder']}\test_df_classification.csv", index=False)

            # Flatten data
            x_train_flatten_channel = train_df.loc[:, list(ml_train_df.columns[2:11])].to_numpy().T
            y_train_flatten_channel = train_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_val_flatten_channel = val_df.loc[:, list(ml_train_df.columns[2:11])].to_numpy().T
            y_val_flatten_channel = val_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_test_flatten_channel = test_df.loc[:, list(test_df.columns[2:11])].to_numpy().T
            y_test_flatten_channel = test_df.loc[:, test_df.columns[-1]].to_numpy().T


        # Create lists to return
        train_pixel_dataset = pixelDatasetGeneration(
            x_train_flatten_channel, y_train_flatten_channel,
            setseed=self.setseed
        )

        val_pixel_dataset = pixelDatasetGeneration(
            x_val_flatten_channel, y_val_flatten_channel,
            setseed=self.setseed
        )

        test_pixel_dataset = pixelDatasetGeneration(
            x_test_flatten_channel, y_test_flatten_channel,
            setseed=self.setseed
        )

        return train_pixel_dataset, val_pixel_dataset, test_pixel_dataset


    def pixel_dataloader_classification(self, coef, batchsize=16*16, num_workers=1):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == 'cuda' else {}

        # Call train, validation, and test pixel dataset
        train_pixel_dataset, val_pixel_dataset, test_pixel_dataset = self.split_data_classification()

        set_seed(self.setseed)
        # FOR TRAIN -----------------------------------
        trainloader = DataLoader(
            dataset=Pixel(train_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            sampler=train_pixel_dataset.weight_and_sampler(coef=coef)[1],
            **kwargs
        )

        set_seed(self.setseed)
        # FOR VALIDATION ------------------------------
        valloader = DataLoader(
            dataset=Pixel(val_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        set_seed(self.setseed)
        # FOR TEST ------------------------------------
        testloader = DataLoader(
            dataset=Pixel(test_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        return trainloader, valloader, testloader, train_pixel_dataset.weight_and_sampler(coef=coef)[0]




    def wercs_oversample(self,
                         X, y,
                         relevance,
                         over,
                         random_state,
                         filters_num=None,
                         filters_value=None,
                         filters=True):

        # Size
        over_size = int(over * len(y))

        # Manipulate X and y
        X, y = np.asarray(X), np.squeeze(np.asarray(y))
        relevance = np.squeeze(np.asarray(relevance))

        # Prob
        if filters:
            relevance_copy = relevance.copy()
            relevance_copy[relevance_copy < filters_num] = filters_value
            prob = np.abs(relevance_copy / np.sum(relevance_copy))
        else:
            prob = np.abs(relevance / np.sum(relevance))

        # Random state
        np.random.seed(seed=random_state)

        # Indexing
        sample_indices = np.random.choice(range(len(y)), size=over_size, p=prob, replace=True)

        # Extra values
        X_over, y_over = X[sample_indices, :], y[sample_indices]

        # New
        X_new = np.append(X_over, X, axis=0)
        y_new = np.append(y_over, y, axis=0)

        return [X_new, y_new]

    def split_data_regression_proportion(self, resample=False):

        # Get train collection
        train_collection = dataCollection(
            self.para_path['train']['general_folder'],
            self.para_path['train']['geometry_domain_list'],
            self.para_path['train']['path_elev'],
            self.para_path['train']['path_wd'],
            self.para_path['train']['path_wse'],
            self.para_path['train']['path_proportion'],
            self.para_path['train']['path_manning'],
            self.para_path['train']['path_roughness'],
            self.para_path['train']['path_sd']
        )

        # Get dataframe for machine learning
        ml_train_df = train_collection.loadpara_into_dataframe_regression_proportion()

        if self.split == True:

            # Split data into train, validation, and test IF SPLIT = TRUE
            train_val_df, test_df = train_test_split(ml_train_df, random_state=42, train_size=0.8, shuffle=True)
            train_df, val_df = train_test_split(train_val_df, random_state=42, train_size=0.75, shuffle=True)

            # Write out just incase
            train_df.to_csv(fr"{self.para_path['train']['general_folder']}\train_df_regression_proportion.csv",
                            index=False)
            val_df.to_csv(fr"{self.para_path['train']['general_folder']}\val_df_regression_proportion.csv", index=False)
            test_df.to_csv(fr"{self.para_path['train']['general_folder']}\test_df_regression_proportion.csv", index=False)

            # Flatten data
            x_train_flatten_channel = train_df.loc[:, list(ml_train_df.columns[2:12])].to_numpy().T
            y_train_flatten_channel = train_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_val_flatten_channel = val_df.loc[:, list(ml_train_df.columns[2:12])].to_numpy().T
            y_val_flatten_channel = val_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_test_flatten_channel = test_df.loc[:, list(ml_train_df.columns[2:12])].to_numpy().T
            y_test_flatten_channel = test_df.loc[:, ml_train_df.columns[-1]].to_numpy().T


        else:
            # Split data into train and validation only IF SPLIT = FALSE
            train_df, val_df = train_test_split(ml_train_df, random_state=42, train_size=0.8, shuffle=True)

            # Get test collection
            test_collection = dataCollection(
                self.para_path['test']['general_folder'],
                self.para_path['test']['geometry_domain_list'],
                self.para_path['test']['path_elev'],
                self.para_path['test']['path_wd'],
                self.para_path['test']['path_wse'],
                self.para_path['test']['path_proportion'],
                self.para_path['test']['path_manning'],
                self.para_path['test']['path_roughness'],
                self.para_path['test']['path_sd']
            )

            # Get dataframe for machine learning
            test_df = test_collection.loadpara_into_dataframe_regression_proportion(name_csv='test')

            # Write out just incase
            train_df.to_csv(fr"{self.para_path['train']['general_folder']}\train_df_regression_proportion.csv", index=False)
            val_df.to_csv(fr"{self.para_path['train']['general_folder']}\val_df_regression_proportion.csv", index=False)
            test_df.to_csv(fr"{self.para_path['test']['general_folder']}\test_df_regression_proportion.csv",
                           index=False)

            # Flatten data
            x_train_flatten_channel = train_df.loc[:, list(ml_train_df.columns[2:12])].to_numpy().T
            y_train_flatten_channel = train_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_val_flatten_channel = val_df.loc[:, list(ml_train_df.columns[2:12])].to_numpy().T
            y_val_flatten_channel = val_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_test_flatten_channel = test_df.loc[:, list(test_df.columns[2:12])].to_numpy().T
            y_test_flatten_channel = test_df.loc[:, test_df.columns[-1]].to_numpy().T


        if resample:
            # Resample
            # FIRST LEVEL -------------
            # Get variables
            x_training_001 = x_train_flatten_channel.transpose().copy()
            y_training_001 = y_train_flatten_channel.copy()
            # Get resampled relevance
            relevance_middle_001 = resreg.sigmoid_relevance(
                y_training_001,
                cl=self.para_path['train']['resample_proportion']['cl_001'],
                ch=self.para_path['train']['resample_proportion']['ch_001']
            ) # 1 is 100% flood, 99 is not 100% flood

            # Plot two-sided relevance values (left and right tails)
            plt.scatter(y_training_001, relevance_middle_001, s=.1)
            plt.ylabel('Relevance')
            plt.xlabel('Target')
            plt.title('Rare domain from both tails')
            plt.show()
            plt.close()
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\relevance_proportion_first_level.jpg",
                bbox_inches='tight', dpi=600
            )

            # Do resampling to create new x and y - first time
            x_train_flatten_channel_resampled_001, y_train_flatten_channel_resampled_001 = self.wercs_oversample(
                x_training_001, y_training_001, relevance_middle_001,
                over=self.para_path['train']['resample_proportion']['over_001'],
                random_state=self.para_path['train']['resample_proportion']['random_state_001'],
                filters_num=self.para_path['train']['resample_proportion']['filters_num_001'],
                filters_value=self.para_path['train']['resample_proportion']['filters_value_001'],
                filters=self.para_path['train']['resample_proportion']['filters_001']
            )

            # Ref: https://stackoverflow.com/questions/176918/how-to-find-the-index-for-a-given-item-in-a-list
            indices_NO_0 = [index for index in range(len(y_train_flatten_channel_resampled_001)) if
                            y_train_flatten_channel_resampled_001[index] != 0]
            y_train_flatten_channel_resampled_NO_0_001 = y_train_flatten_channel_resampled_001[indices_NO_0]
            x_train_flatten_channel_resampled_NO_0_001 = x_train_flatten_channel_resampled_001[indices_NO_0]

            # Plot
            plt.hist(y_train_flatten_channel_resampled_NO_0_001, bins=100)
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\resample_proportion_first_level.jpg",
                bbox_inches='tight', dpi=600
            )


            # SECOND LEVEL -------------
            x_training_002 = x_train_flatten_channel_resampled_NO_0_001.copy()
            y_training_002 = y_train_flatten_channel_resampled_NO_0_001.copy()
            relevance_middle_002 = resreg.sigmoid_relevance(
                y_training_002,
                cl=self.para_path['train']['resample_proportion']['cl_002'],
                ch=self.para_path['train']['resample_proportion']['ch_002']
            )

            # Plot two-sided relevance values (left and right tails)
            plt.scatter(y_training_002, relevance_middle_002, s=.1)
            # plt.axhline(.48, linestyle='--', color='black')
            plt.ylabel('Relevance')
            plt.xlabel('Target')
            plt.title('Rare domain from both tails')
            plt.show()
            plt.close()
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\relevance_proportion_second_level.jpg",
                bbox_inches='tight', dpi=600
            )

            # Do resampling to create new x and y - second time
            x_train_flatten_channel_resampled_002, y_train_flatten_channel_resampled_002 = self.wercs_oversample(
                x_training_002,
                y_training_002,
                relevance_middle_002,
                over=self.para_path['train']['resample_proportion']['over_002'],
                random_state=self.para_path['train']['resample_proportion']['random_state_002'],
                filters_num=self.para_path['train']['resample_proportion']['filters_num_002'],
                filters_value=self.para_path['train']['resample_proportion']['filters_value_002'],
                filters=self.para_path['train']['resample_proportion']['filters_002']
            )

            # Plot
            plt.hist(y_train_flatten_channel_resampled_002, bins=100)
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\resample_proportion_second_level.jpg",
                bbox_inches='tight', dpi=600
            )

            # Extract 0
            indices_0 = [index for index in range(len(y_train_flatten_channel_resampled_001)) if
                         y_train_flatten_channel_resampled_001[index] == 0]
            y_train_flatten_channel_resampled_0_002 = y_train_flatten_channel_resampled_001[indices_0]
            x_train_flatten_channel_resampled_0_002 = x_train_flatten_channel_resampled_001[indices_0]

            # Adding back to 0
            x_train_flatten_channel_full = np.append(x_train_flatten_channel_resampled_0_002,
                                                     x_train_flatten_channel_resampled_002, axis=0)
            y_train_flatten_channel_full = np.append(y_train_flatten_channel_resampled_0_002,
                                                     y_train_flatten_channel_resampled_002, axis=0)


            # FULL --------
            plt.hist(y_train_flatten_channel_full, bins=100)
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\resample_proportion_full.jpg",
                bbox_inches='tight', dpi=600
            )

            x_train_flatten_channel_R = x_train_flatten_channel_full.T.copy()
            y_train_flatten_channel_R = y_train_flatten_channel_full.copy()

            # Create lists to return
            train_pixel_dataset = pixelDatasetGeneration(
                x_train_flatten_channel_R, y_train_flatten_channel_R,
                setseed=self.setseed
            )

            val_pixel_dataset = pixelDatasetGeneration(
                x_val_flatten_channel, y_val_flatten_channel,
                setseed=self.setseed
            )

            test_pixel_dataset = pixelDatasetGeneration(
                x_test_flatten_channel, y_test_flatten_channel,
                setseed=self.setseed
            )

            return train_pixel_dataset, val_pixel_dataset, test_pixel_dataset


        else:


            # Create lists to return
            train_pixel_dataset = pixelDatasetGeneration(
                x_train_flatten_channel, y_train_flatten_channel,
                setseed=self.setseed
            )

            val_pixel_dataset = pixelDatasetGeneration(
                x_val_flatten_channel, y_val_flatten_channel,
                setseed=self.setseed
            )

            test_pixel_dataset = pixelDatasetGeneration(
                x_test_flatten_channel, y_test_flatten_channel,
                setseed=self.setseed
            )

            return train_pixel_dataset, val_pixel_dataset, test_pixel_dataset



    def pixel_dataloader_regression_proportion(self,
                                               batchsize=3200,
                                               num_workers=1,
                                               resample=False):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == 'cuda' else {}

        # Call train, validation, and test pixel dataset
        train_pixel_dataset, val_pixel_dataset, test_pixel_dataset = self.split_data_regression_proportion(
            resample=resample)

        set_seed(self.setseed)
        # FOR TRAIN -----------------------------------
        trainloader = DataLoader(
            dataset=Pixel(train_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        set_seed(self.setseed)
        # FOR VALIDATION ------------------------------
        valloader = DataLoader(
            dataset=Pixel(val_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        set_seed(self.setseed)
        # FOR TEST ------------------------------------
        testloader = DataLoader(
            dataset=Pixel(test_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        return trainloader, valloader, testloader







    def split_data_regression_sd(self, resample=False):

        # Get train collection
        train_collection = dataCollection(
            self.para_path['train']['general_folder'],
            self.para_path['train']['geometry_domain_list'],
            self.para_path['train']['path_elev'],
            self.para_path['train']['path_wd'],
            self.para_path['train']['path_wse'],
            self.para_path['train']['path_proportion'],
            self.para_path['train']['path_manning'],
            self.para_path['train']['path_roughness'],
            self.para_path['train']['path_sd']
        )

        # Get dataframe for machine learning
        ml_train_df = train_collection.loadpara_into_dataframe_regression_sd()

        if self.split == True:

            # Split data into train, validation, and test IF SPLIT = TRUE
            train_val_df, test_df = train_test_split(ml_train_df, random_state=42, train_size=0.8, shuffle=True)
            train_df, val_df = train_test_split(train_val_df, random_state=42, train_size=0.75, shuffle=True)

            # Write out just incase
            train_df.to_csv(fr"{self.para_path['train']['general_folder']}\train_df_regression_sd.csv",
                            index=False)
            val_df.to_csv(fr"{self.para_path['train']['general_folder']}\val_df_regression_sd.csv", index=False)
            test_df.to_csv(fr"{self.para_path['train']['general_folder']}\test_df_regression_sd.csv", index=False)

            # Flatten data
            x_train_flatten_channel = train_df.loc[:, list(ml_train_df.columns[2:10])].to_numpy().T
            y_train_flatten_channel = train_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_val_flatten_channel = val_df.loc[:, list(ml_train_df.columns[2:10])].to_numpy().T
            y_val_flatten_channel = val_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_test_flatten_channel = test_df.loc[:, list(ml_train_df.columns[2:10])].to_numpy().T
            y_test_flatten_channel = test_df.loc[:, ml_train_df.columns[-1]].to_numpy().T


        else:
            # Split data into train and validation only IF SPLIT = FALSE
            train_df, val_df = train_test_split(ml_train_df, random_state=42, train_size=0.8, shuffle=True)

            # Get test collection
            test_collection = dataCollection(
                self.para_path['test']['general_folder'],
                self.para_path['test']['geometry_domain_list'],
                self.para_path['test']['path_elev'],
                self.para_path['test']['path_wd'],
                self.para_path['test']['path_wse'],
                self.para_path['test']['path_proportion'],
                self.para_path['test']['path_manning'],
                self.para_path['test']['path_roughness'],
                self.para_path['test']['path_sd']
            )

            # Get dataframe for machine learning
            test_df = test_collection.loadpara_into_dataframe_regression_sd(name_csv='test')

            # Write out just incase
            train_df.to_csv(fr"{self.para_path['train']['general_folder']}\train_df_regression_sd.csv", index=False)
            val_df.to_csv(fr"{self.para_path['train']['general_folder']}\val_df_regression_sd.csv", index=False)
            test_df.to_csv(fr"{self.para_path['test']['general_folder']}\test_df_regression_sd.csv", index=False)

            # Flatten data
            x_train_flatten_channel = train_df.loc[:, list(ml_train_df.columns[2:10])].to_numpy().T
            y_train_flatten_channel = train_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_val_flatten_channel = val_df.loc[:, list(ml_train_df.columns[2:10])].to_numpy().T
            y_val_flatten_channel = val_df.loc[:, ml_train_df.columns[-1]].to_numpy().T

            x_test_flatten_channel = test_df.loc[:, list(test_df.columns[2:10])].to_numpy().T
            y_test_flatten_channel = test_df.loc[:, test_df.columns[-1]].to_numpy().T


        if resample:
            # Resample
            # FIRST LEVEL -------------
            # Get variables
            x_training = x_train_flatten_channel.transpose().copy()
            y_training = y_train_flatten_channel.copy()
            # Get resampled relevance
            relevance_middle = resreg.sigmoid_relevance(
                y_training,
                cl=self.para_path['train']['resample_sd']['cl'],
                ch=self.para_path['train']['resample_sd']['ch'])

            # Plot two-sided relevance values (left and right tails)
            plt.scatter(y_training, relevance_middle, s=.1)
            plt.ylabel('Relevance')
            plt.xlabel('Target')
            plt.title('Rare domain from both tails')
            plt.show()
            plt.close()
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\relevance_sd_level.jpg",
                bbox_inches='tight', dpi=600
            )

            plt.hist(y_training, bins=100)
            plt.show()
            plt.close()
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\distribution_full_sd_level.jpg",
                bbox_inches='tight', dpi=600
            )

            plt.hist(y_training[y_training != 0], bins=100)
            plt.show()
            plt.close()
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\distribution_no0_sd_level.jpg",
                bbox_inches='tight', dpi=600
            )

            # Do resampling to create new x and y - first time
            x_train_flatten_channel_resampled, y_train_flatten_channel_resampled = resreg.wercs(
                x_training, y_training, relevance_middle,
                over=self.para_path['train']['resample_sd']['over'],
                under=self.para_path['train']['resample_sd']['under'],
                noise=self.para_path['train']['resample_sd']['noise'],
                random_state=self.para_path['train']['resample_sd']['random_state']
            )

            x_train_flatten_channel_R = x_train_flatten_channel_resampled.T.copy()
            y_train_flatten_channel_R = y_train_flatten_channel_resampled.copy()

            plt.hist(y_train_flatten_channel_R, bins=50)
            plt.show()
            plt.close()
            # Save fig
            plt.savefig(
                fr"{self.para_path['train']['general_folder']}\distribution_afterresampling_sd_level.jpg",
                bbox_inches='tight', dpi=600
            )

            # Create lists to return
            train_pixel_dataset = pixelDatasetGeneration(
                x_train_flatten_channel_R, y_train_flatten_channel_R,
                setseed=self.setseed
            )

            val_pixel_dataset = pixelDatasetGeneration(
                x_val_flatten_channel, y_val_flatten_channel,
                setseed=self.setseed
            )

            test_pixel_dataset = pixelDatasetGeneration(
                x_test_flatten_channel, y_test_flatten_channel,
                setseed=self.setseed
            )

            return train_pixel_dataset, val_pixel_dataset, test_pixel_dataset


        else:


            # Create lists to return
            train_pixel_dataset = pixelDatasetGeneration(
                x_train_flatten_channel, y_train_flatten_channel,
                setseed=self.setseed
            )

            val_pixel_dataset = pixelDatasetGeneration(
                x_val_flatten_channel, y_val_flatten_channel,
                setseed=self.setseed
            )

            test_pixel_dataset = pixelDatasetGeneration(
                x_test_flatten_channel, y_test_flatten_channel,
                setseed=self.setseed
            )

            return train_pixel_dataset, val_pixel_dataset, test_pixel_dataset



    def pixel_dataloader_regression_sd(self,
                                       batchsize=3200,
                                       num_workers=1,
                                       resample=False):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == 'cuda' else {}

        # Call train, validation, and test pixel dataset
        train_pixel_dataset, val_pixel_dataset, test_pixel_dataset = self.split_data_regression_sd(
            resample=resample)

        set_seed(self.setseed)
        # FOR TRAIN -----------------------------------
        trainloader = DataLoader(
            dataset=Pixel(train_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        set_seed(self.setseed)
        # FOR VALIDATION ------------------------------
        valloader = DataLoader(
            dataset=Pixel(val_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        set_seed(self.setseed)
        # FOR TEST ------------------------------------
        testloader = DataLoader(
            dataset=Pixel(test_pixel_dataset.pixel_dataset_generation()),
            batch_size=batchsize,
            shuffle=False,
            **kwargs
        )

        return trainloader, valloader, testloader







