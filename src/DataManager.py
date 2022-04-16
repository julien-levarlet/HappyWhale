# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from src.WhaleDataset import WhaleDataset
from src.transformation import Transformation

class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, annotations_file : str , 
                dataFolderPath : str, 
                batch_size: int = 1,
                test_percentage : float = 0.2, 
                val_percentage : float = 0.2, 
                transform_proba=0.8,
                img_size=256,
                verbose=False, 
                **kwargs):
        """
        Args:
            annotations_file (str): CSV file with 2 columns :  * 'image' witch contains image name (imgname.jpg)
                                                        * 'individual_id' witch contains the id of the animal
            dataFolderPath (str): Path to the folder containing all the images
            batch_size (int, optional): Batch size. Defaults to 1.
            test_percentage (float, optional): Percentage of images used for test. Defaults to 0.2.
            val_percentage (float, optional): Percentage of images used for validation on all the images that are not used for testing (1- test_percentage). Defaults to 0.2.
            transform_proba (float, optional): Probability of doing data augmentation. Defaults to 0.5.
            img_size (int, optional): Size of the images before entering the neural net. Defaults to 256.
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

        # separation train/test/validation sets
        df_train_val, df_test = train_test_split(df_labels, test_size=test_percentage, stratify=df_labels["individual_id"])
        df_train, df_val = train_test_split(df_train_val, test_size=val_percentage, stratify=df_train_val["individual_id"])

        if verbose:
            print(df_labels.head(5))
            print("Dataset size :", len(df_labels))
            print("Size of validation set :", len(df_val))
            print("Size of test set :", len(df_test))
            print("Size of train set :", len(df_train))

        # data augmentation class
        tranform = Transformation(image_size=(img_size, img_size)).apply_transformations

        # creates Datasets and DataLoaders
        self.train_set = WhaleDataset(df_train, dataFolderPath, transform_proba=transform_proba, img_size=img_size, transform=tranform)
        self.val_set = WhaleDataset(df_val, dataFolderPath, transform_proba=transform_proba, img_size=img_size, transform=tranform)
        self.test_set = WhaleDataset(df_test, dataFolderPath,transform_proba=transform_proba, img_size=img_size, transform=tranform)

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