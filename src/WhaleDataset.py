import os
from random import Random
import pandas as pd
import torchvision, torch
import cv2
from torch.utils import data

class WhaleDataset(data.Dataset):
    '''
    Redefinition of torch.utils.data.dataset for the csv used in the HappyWhale Challenge
    '''

    def __init__(self, dataset : pd.DataFrame,
                 img_dir, transform=None):
        """
        Args:
            dataset (pd.DataFrame): Dataset, with images name (column 0) and their ids (column 1)
            img_dir (str): Path to the folder containing all the images
            transform (Callable, optional): Function used to do the data augmentation. Defaults to None for no augmentation.
        """

        self.proba = 0.5 # probability of using data augmentation on an image
        self.img_df = dataset
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img = self.img_df.iat[idx,0]
        label = self.img_df.iat[idx,1]

        img_path = os.path.join(self.img_dir, img)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError("The image does not exists: image name=",img_path)

        if self.transform and self.p < Random.random():
            image = self.transform(image)
        
        image = torchvision.transforms.functional.to_tensor(image)
        label = torch.tensor(label)
        return image, label