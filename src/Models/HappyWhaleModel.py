"""
Inspired from :
    Debarshi Chanda
    https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter/notebook
"""

import torch.nn as nn
import torch
import timm

from src.Models.ArcFaceMarginProduct import ArcMarginProduct

class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size, num_class, arcface_config):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.classifier.in_features
        print(in_features)
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, 
                                   num_class,
                                   s=arcface_config["s"], 
                                   m=arcface_config["m"], 
                                   easy_margin=arcface_config["easy_margin"], 
                                   ls_eps=arcface_config["ls_eps"])

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding, labels)
        return output
    
    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'