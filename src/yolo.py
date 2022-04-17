import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
import os


class Yolo_Transformation():

    """
    use Yolo on your image to crop 
    display analyse on image class recognition
    propose multi recognition solution
    """

    def __init__(self,loading_folder,saving_folder,tab_image_name,model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)):
        
        """
        arguments:
        model=model from ultralytics/yolov5 to use
        loading_folder= folder which contains all of your image to crop with yolo
        saving_folder= folder which will contain all of your image croped
        tab_image_name= np.array or list which contain the name of your image to crop
        """

        self.model=model
        self.loading_folder=loading_folder
        self.saving_folder=saving_folder
        self.tab_image_name=tab_image_name


    
    #load image with cv2
    def load_image(self,filename):
        return cv2.imread(filename)

    #save image with cv2
    def save_image(self,filename,image):
        return cv2.imwrite(filename, image)

    def use_yolo(self):
        imgs=[]
        for k in range(len(self.tab_image_name)):
            imgs.append(self.load_image(self.loading_folder+self.tab_image_name[k]))
        return self.model(imgs)

    #crop image in saving_folder using tab_image_name with yolo
    def crop_image_from_yolo(self,yolo_results):
        for k in range(len(yolo_results)):
            working_prediction=yolo_results.pandas().xyxy[k]
            filename=self.loading_folder+self.tab_image_name[k]
            if(working_prediction.shape[0]==1):
                #print("easy to deal with: ",self.tab_image_name[k])
                image=self.load_image(filename)
                image_croped=self.crop_image(image,working_prediction.iloc[0,0],working_prediction.iloc[0,1],working_prediction.iloc[0,2],working_prediction.iloc[0,3])
                self.save_image(self.saving_folder+self.tab_image_name[k],image_croped)
            else:
                #print("hard to deal with: ",self.tab_image_name[k])
                image=self.load_image(filename)
                self.save_image(self.saving_folder+self.tab_image_name[k],image)

    def crop_image(self,img: np.ndarray, x_min:int, y_min:int, x_max:int, y_max:int):
        """Crop an image based on bounding box coordinate in pixels
        (0,0) correspond to the top left hand corner 

        Args:
            img (np.ndarray): the image to crop
            x_min (int): min abscissa of the bounding box
            y_min (int): min ordinate
            x_max (int): max abscissa
            y_max (int): max ordinate

        Returns:
            np.ndarray: the crop of the image
        """
        return img[ int(y_min):int(y_max),int(x_min):int(x_max)]

    # analyse image class recognition to propose multi recognition solution
    def analyse_yolo_result(self,yolo_results):
        count_one_prediction_image=0
        count_multi_prediction_image=0
        count_no_prediction_image=0
        find_dauphin_as=[]
        ponderation=[]
        confidence=[]
        find_something_as=[]
        for k in range(len(yolo_results)):
            working_prediction=yolo_results.pandas().xyxy[k]
            if(working_prediction.shape[0]==1):
                count_one_prediction_image+=1
                flag=True
                for j in range(len(find_dauphin_as)):
                    if(working_prediction["name"].iloc[0]==find_dauphin_as[j]):
                        ponderation[j]+=1
                        confidence[j]+=working_prediction["confidence"].iloc[0]
                        flag=False
                if(flag==True):
                    find_dauphin_as.append(working_prediction["name"].iloc[0])
                    ponderation.append(1)
                    confidence.append(working_prediction["confidence"].iloc[0])
            else:
                list=[]
                for j in range(working_prediction.shape[0]):
                    list.append(working_prediction["name"].iloc[j])
                if(list!=[]):
                    count_multi_prediction_image+=1
                    find_something_as.append(list)
                else:
                    count_no_prediction_image+=1
        return [count_one_prediction_image,count_multi_prediction_image,find_dauphin_as,ponderation,confidence,find_something_as,count_no_prediction_image]

    #display analyse metrics
    def display_yolo_analyse(self,analyses):
        tab=pd.DataFrame(np.array([analyses[2],analyses[3],analyses[4]]).T,columns=["prediction","ponderation","confidence"])
        print("yolo croped sumary:")
        print("number of crop maked by single prediction: ",analyses[0])
        print("number of image ignore because of no prediction: ",analyses[6])
        print("number of image ignore because of ambigous prediction: ",analyses[1])
        print("dolphin can be predict as:")
        print(tab)
        print("ambigous prediction are:")
        print(analyses[5])
        
