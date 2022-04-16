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
        '''
        Input :
        - dataset : dataset, with images name (column 0) and their ids (column 1)
        - img_dir : path to the folder containing all the images
        - set : dataset searched, train, validation or test                                                
        '''
        self.proba = 0.5
        self.img_list = dataset
        self.img_dir = img_dir
        self.transform_tensor = torchvision.transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list.iat[idx,0]
        label = self.img_list.iat[idx,1]

        img_path = os.path.join(self.img_dir, img)
        image = cv2.imread(img_path)

        if self.transform and self.p < Random.random():
            image = self.transform(image)
        image = self.transform_tensor(image)
        label = torch.tensor(label)
        return image, label