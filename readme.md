# HappyWhale Kaggle Challenge - Team WhalePlayed

## Introduction

This projet was made for the IFT 780 course at Université de Sherbrooke. 

We use fingerprints and facial recognition to identify people, but can we use similar approaches with marine mammals?

You will find on this repository the work done by Gaétan Rey, Julien Levarlet, Timothée Wright and Firas Abouda, for the challenge https://www.kaggle.com/competitions/happy-whale-and-dolphin/overview 



## Installation :

To install the dependancies run : `pip install requirements.txt`

Dataset can be obtained at :
https://drive.google.com/drive/folders/1xsIEZkm7YXptVXHMPTJKNtZIkBw9F37b?usp=sharing

The dataset with the images of the most common classes should be located in the data folder and named "common_cropped_train_imgs" and the CSV files should be in the data folder too. 

## Code

In this projet, we used 2 notebooks :
- [HappyWhale](HappyWhale.ipynb) which is used to test code locally.
- [kaggle](kaggle.ipynb) which is the notebook used to do the submission on Kaggle for the competition.

The [src](src) folder contains all the code done during the project.<br>
The file [DataSampler](src/DataSampler.py) was used to create our databases accessible on the Google drive to test our architectures.