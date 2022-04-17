import os
import random
import pandas as pd
import torchvision, torch
import cv2
from torch.utils import data
from src.utils import image_resize

class WhaleDataset(data.Dataset):
    '''
    Redefinition of torch.utils.data.dataset for the csv used in the HappyWhale Challenge
    '''

    def __init__(self, dataset : pd.DataFrame,
                 img_dir,img_size=256, transform=None, transform_proba=0.5):
        """
        Args:
            dataset (pd.DataFrame): Dataset, with images name (column 0) and their ids (column 1)
            img_dir (str): Path to the folder containing all the images
            transform (Callable, optional): Function used to do the data augmentation. Defaults to None for no augmentation.
        """
        self.img_size = img_size
        self.proba = transform_proba # probability of using data augmentation on an image
        self.img_df = dataset
        self.img_dir = img_dir
        self.transform = transform
        self.individual_column = self.img_df.columns.get_loc("individual_id")
        self.img_column = self.img_df.columns.get_loc("image")
        print(self.individual_column, self.img_column)

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img = self.img_df.iat[idx,self.img_column]
        label = self.img_df.iat[idx,self.individual_column]

        img_path = os.path.join(self.img_dir, img)
        image = image_resize(cv2.imread(img_path), self.img_size)
        if image is None:
            raise FileNotFoundError("The image does not exists: image name=",img_path)

        if self.transform is not None and self.proba < random.random():
            image = self.transform(image)[1]
        image = torchvision.transforms.functional.to_tensor(image)
        label = torch.tensor(label,dtype=torch.long)
        return image, label