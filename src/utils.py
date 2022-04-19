"""
This is used to write fonction that can useful in several part of the project
"""
import cv2
import numpy as np
import torch


def image_resize(img: np.ndarray, dim: int, pad: bool=False)->np.ndarray:
    """Image resizing function, it can resize the image by squezzing the image, 
    or by padding the image before resizing to preserve the original ratio.
    It does not a change the original image.

    Args:
        img (np.ndarray): the image to be resize
        dim (Tuple[int,int]): the dimention wanted
        pad (bool, optional): use the padding or not. Defaults to False.

    Returns:
        np.ndarray: the image resized
    """
    if pad:
        temp_size = max(img.shape)
        x_pad = (temp_size-img.shape[0])// 2
        y_pad = (temp_size-img.shape[1])// 2
        pad_size = [[x_pad], [y_pad], [0]]
        img = np.pad(img, pad_size)
    return cv2.resize(img, (dim,dim))

def crop_image(img: np.ndarray, x_min:int, y_min:int, x_max:int, y_max:int):
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
    return img[x_min:x_max, y_min:y_max]

def accuracy(input: torch.Tensor, target: torch.Tensor)-> float:
    """Compute the accuracy of the model

    Args:
        input (torch.Tensor): predicted outputs of the model
        target (torch.Tensor): targets

    Returns:
        torch.Tensor: tensor containing the accuracy
    """
    #print("INPUT ACCURACY = ",input)
    predicted = input.argmax(dim=1)
    #print("predicted =", predicted)
    #print("target =", target)
    #print("diff =", predicted == target)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

