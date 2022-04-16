# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from src.WhaleDataset import WhaleDataset

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
        #df_labels = df_labels.head(100)

        # separation train/test/validation sets
        val_test_sep = int(df_labels.shape[0] * (1-test_percentage))
        train_val_sep = int(df_labels.shape[0] * (1-test_percentage) * (1-val_percentage) )

        test_data = df_labels[val_test_sep:]
        val_data = df_labels[train_val_sep:val_test_sep]
        train_data = df_labels[:train_val_sep]

        df_train_val, df_test = train_test_split(df_labels, test_size=test_percentage, stratify=df_labels["individual_id"])
        df_train, df_val = train_test_split(df_train_val, test_size=val_percentage, stratify=df_train_val["individual_id"])

        if verbose:
            print(df_labels.head(5))
            print("Dataset size :", len(df_labels))
            print("Size of validation set :", len(df_val))
            print("Size of test set :", len(df_test))
            print("Size of train set :", len(df_train))

        # creates Datasets and DataLoaders
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