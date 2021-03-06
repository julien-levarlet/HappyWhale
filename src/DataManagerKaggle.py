# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.WhaleDataset import WhaleDataset
from src.transformation import Transformation

class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data, used for submission data on Kaggle
    """

    def __init__(self, annotations_file : str ,
                test_file : str, 
                dataFolderPath : str, 
                batch_size: int = 1,
                val_percentage : float = 0.2, 
                transform_proba=0.8,
                img_size=256,
                verbose=False, 
                **kwargs):
        """
        Args:
            annotations_file (str): CSV file with 2 columns :  * 'image' witch contains image name (imgname.jpg)
                                                        * 'individual_id' witch contains the id of the animal
            test_file (str): CSV file with 2 columns : contains default values of submission file on kaggle
            dataFolderPath (str): Path to the folder containing both training and testing folder
            batch_size (int, optional): Batch size. Defaults to 1.
            val_percentage (float, optional): Percentage of images used for validation. Defaults to 0.2.
            transform_proba (float, optional): Probability of doing data augmentation. Defaults to 0.8.
            img_size (int, optional): Size of the images before entering the neural net. Defaults to 256.
            verbose (bool, optional): Display information on datasets. Defaults to False.
        """
        self.batch_size = batch_size
        self.dataFolderPath = dataFolderPath

        # reads data
        df_labels = pd.read_csv(annotations_file)
        self.df_test = pd.read_csv(test_file)
        
        self.df_test["individual_id"] = -1
        

        # labels need to be categorical 
        self.encoder = LabelEncoder()
        df_labels['individual_id'] = self.encoder.fit_transform(df_labels['individual_id'])

        # permutation
        df_labels.sample(frac=1)

        # separation train/test/validation sets
        self.df_train, self.df_val = train_test_split(df_labels, test_size=val_percentage)

        if verbose:
            print(df_labels.head(5))
            print("Dataset size :", len(df_labels))
            print("Size of validation set :", len(self.df_val))
            print("Size of test set :", len(self.df_test))
            print("Size of train set :", len(self.df_train))

        # data augmentation class
        tranform = Transformation(image_size=(img_size, img_size)).apply_transformations

        # creates Datasets and DataLoaders
        self.train_set = WhaleDataset(self.df_train, dataFolderPath + "/cropped_train_images/cropped_train_images/", transform_proba=transform_proba, img_size=img_size, transform=tranform)
        self.val_set = WhaleDataset(self.df_val, dataFolderPath + "/cropped_train_images/cropped_train_images/", transform_proba=transform_proba, img_size=img_size, transform=tranform)
        self.test_set = WhaleDataset(self.df_test, dataFolderPath + "/cropped_test_images/cropped_test_images/",transform_proba=transform_proba, img_size=img_size, transform=tranform)

        self.train_loader = DataLoader(self.train_set, batch_size, num_workers=4, shuffle=True, **kwargs)
        self.validation_loader = DataLoader(self.val_set, batch_size, num_workers=4, shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_set, batch_size, num_workers=4, shuffle=True, **kwargs)


    def get_train_set(self):
        return self.df_train

    def get_validation_set(self):
        return self.df_val
    
    def get_test_set(self):
        return self.df_test

    def get_train_loader(self):
        return self.train_loader

    def get_validation_loader(self):
        return self.validation_loader

    def get_test_loader(self):
        return self.test_loader

    def get_batch_size(self):
        return self.batch_size

    def get_random_sample_from_test_set(self):
        indice = np.random.randint(0, len(self.test_set))
        return self.test_set[indice]

    def get_number_of_classes(self):
        return len(self.encoder.classes_)

    def get_individual_id(self, class_numbers:np.ndarray):
        """Transform number labels to individual ids

        Args:
            class_numbers (np.ndarray): the array of number labels

        Returns:
            np.ndarray: the array of individual ids
        """
        return self.encoder.inverse_transform(class_numbers)
