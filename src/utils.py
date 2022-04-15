"""
This is used to write fonction that can useful in several part of the project
"""
from typing import Tuple
import cv2
import numpy as np


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