"""
File adapted from :
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from os.path import join

class BaseModel(nn.Module):

    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        self.threshold = 0.3 # threshold value to detect new individuals
    
    def save(self, exp_name):
        """
        Save the model's checkpoint into the filename
        :arg
            filename: file in which to save the model
        """
        filename = exp_name + "_" + self.__class__.__name__ + '.pt'
        torch.save(self.state_dict(), join("./save/", filename))

    def load(self, exp_name):
        """
        Load the model's weights saved into the filename
        :arg
            file_path: path file where model's weights are saved
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        path = "./save"
        filename = exp_name + "_" + self.__class__.__name__ + '.pt'
        self.load_state_dict(torch.load(join(path, filename), map_location="cpu"))