# -*- coding:utf-8 -*-

import os
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets
import pandas as pd

from WhaleDataset import WhaleDataset

class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, annotations_file : str , dataFolderPath : str, batch_size: int = 1,
                test_percentage : float = 0.2, val_percentage : float = 0.2, verbose=False, **kwargs):
        """
        Args:
            annotations_file (str): CSV file with 2 columns :  * 'image' witch contains image name (imgname.jpg)
                                                        * 'individual_id' witch contains the id of the animal
            dataFolderPath (str): Path to the folder containing all the images
            batch_size (int, optional): Batch size. Defaults to 1.
            test_percentage (float, optional): Percentage of images used for test. Defaults to 0.2.
            val_percentage (float, optional): Percentage of images used for validation on all the images that are not used for testing (1- test_percentage). Defaults to 0.2.
            verbose (bool, optional): Display information on datasets. Defaults to False.
        """
        self.batch_size = batch_size
        self.dataFolderPath = dataFolderPath

        # reads data
        df_labels = pd.read_csv(annotations_file)
        # labels need to be categorical 
        df_labels["individual_id"] = df_labels["individual_id"].astype('category').cat.codes

        # permutation
        df_labels.sample(frac=1)
        df_labels = df_labels.head(100)

        # separation train/test/validation sets
        val_test_sep = int(df_labels.shape[0] * (1-test_percentage))
        train_val_sep = int(df_labels.shape[0] * (1-test_percentage) * (1-val_percentage) )

        test_data = df_labels[val_test_sep:]
        val_data = df_labels[train_val_sep:val_test_sep]
        train_data = df_labels[:train_val_sep]
        if verbose:
            print(df_labels.head(5))
            print("Dataset size :", len(df_labels))
            print("Size of validation set :", len(val_data))
            print("Size of test set :", len(test_data))
            print("Size of train set :", len(train_data))

        # creates Datasets and DataLoaders
        self.train_set = WhaleDataset(train_data, dataFolderPath)
        self.val_set = WhaleDataset(val_data, dataFolderPath)
        self.test_set = WhaleDataset(test_data, dataFolderPath)

        self.train_loader = DataLoader(self.train_set, batch_size, shuffle=True, **kwargs)
        self.validation_loader = DataLoader(self.val_set, batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_set, batch_size, shuffle=True, **kwargs)


    def get_train_set(self):
        return self.train_loader

    def get_validation_set(self):
        return self.validation_loader

    def get_test_set(self):
        return self.test_loader

    def get_batch_size(self):
        return self.batch_size

    def get_random_sample_from_test_set(self):
        indice = np.random.randint(0, len(self.test_set))
        return self.test_set[indice]