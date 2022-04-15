import os
import pandas as pd
from torchvision.io import read_image
from torch.utils import data

class CustomImageDataset(data.Dataset):
    def __init__(self, annotations_file : str, img_dir : str, set : str, 
                test_percentage : float = 0.2, val_percentage : float = 0.2,
                transform=None, target_transform=None):
        '''
        Input :
        - annotations_file : CSV file with 2 columns :  * 'image' witch contains image name (imgname.jpg)
                                                        * 'individual_id' witch contains the id of the animal
        - img_dir : path to the folder containing all the images
        - set : dataset searched, train, validation or test                                                
        
        '''
        df_labels = pd.read_csv(annotations_file)
        df_labels.sample(frac=1)


        separation = int(df_labels.shape[0] * 0.8)

        df_labels[:separation]

        self.img_list = 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        row = self.img_list.iloc[[0]]
        img_path = os.path.join(self.img_dir, row['image'])
        image = read_image(img_path)
        label = row['individual_id']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label