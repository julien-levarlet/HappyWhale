# -*- coding:utf-8 -*-

import os
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from WhaleDataset import WhaleDataset

class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, annotations_file : str , dataFolderPath : str, batch_size: int = 1,
                test_percentage : float = 0.2, val_percentage : float = 0.2, **kwargs):
        """
        Input :
        - annotations_file : CSV file with 2 columns :  * 'image' witch contains image name (imgname.jpg)
                                                        * 'individual_id' witch contains the id of the animal
        - dataFolderPath : path to the folder containing all the images
        - batch_size : batch_size
        - test_percentage : percentage of images used for test
        - val_percentage : percentage of images used for validation on all the images that are not used for testing (1- test_percentage)
        """
        self.batch_size = batch_size
        self.dataFolderPath = dataFolderPath

        df_labels = pd.read_csv(annotations_file)
        # labels need to be categorical 
        df_labels["individual_id"] = df_labels["individual_id"].astype('category').cat.codes

        df_train_val, df_test = train_test_split(df_labels, test_size=test_percentage, stratify=df_labels["individual_id"])

        df_train, df_val = train_test_split(df_train_val, test_size=val_percentage, stratify=df_train_val["individual_id"])

        self.train_set = WhaleDataset(df_train, dataFolderPath)
        self.val_set = WhaleDataset(df_val, dataFolderPath)
        self.test_set = WhaleDataset(df_test, dataFolderPath)

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