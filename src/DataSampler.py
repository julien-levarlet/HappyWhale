# -*- coding:utf-8 -*-

"""
This code sample images from the global database to extract somes classes from experiment purposes
"""

import pandas as pd
from shutil import copyfile
import os
import csv

nb_classes = 20
max_img_per_class = 50

file_source = 'dataset/cropped_train_images'
file_destination = 'dataset/common_cropped_train_imgs'

def loadData():

    df = pd.read_csv('dataset/train2.csv',
        sep=',',
        usecols=['individual_id', 'image'],
        dtype={
            'individual_id' : str, 'image' : str}
    )

    value_count = df['individual_id'].value_counts()

    # we select min_nb_of_img for each nb_classes classes. So each class has the same number of elements
    min_nb_of_img = min(value_count.values.tolist()[nb_classes], max_img_per_class)

    most_common_ids = value_count.index.tolist()[:nb_classes]

    img_most_common_classes = df.loc[df['individual_id'].isin(most_common_ids)]

    # reduce number of images per class to min_nb_of_img
    sampled_most_common_classes = img_most_common_classes.groupby('individual_id').head(min_nb_of_img)

    # Create a CSV containing only the selected images
    sampled_most_common_classes.to_csv(path_or_buf='dataset/common_train.csv', index=False)

    # Copy all the selected images to another folder
    for file in sampled_most_common_classes['image']:
        copyfile(os.path.join(file_source, file), os.path.join(file_destination, file))
    
    print("done")


if __name__ == "__main__":
    loadData()