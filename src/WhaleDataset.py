import os
import pandas as pd
from torchvision.io import read_image
from torch.utils import data

class WhaleDataset(data.Dataset):
    def __init__(self, dataset : data.Dataset,
                transform=None, target_transform=None):
        '''
        Input :
        - annotations_file : CSV file with 2 columns :  * 'image' witch contains image name (imgname.jpg)
                                                        * 'individual_id' witch contains the id of the animal
        - img_dir : path to the folder containing all the images
        - set : dataset searched, train, validation or test                                                
        '''

        self.img_list = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        row = self.img_list.iloc[[idx]]
        img_path = os.path.join(self.img_dir, row['image'])
        image = read_image(img_path)
        label = row['individual_id']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label