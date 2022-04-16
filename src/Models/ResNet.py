# developper dans le cadre du projet kaggle happywhale en Avril 2022
# auteurs: Rey Gaetan, Wright Timothee, Levarlet Julien, Firas Abouda

# cette architecture resnet est notre propre implementation du papier de recherche suivant: https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8

import torch.nn as nn
from src.Models.ArcFace import ArcMarginProduct


class ResNet(nn.Module):

    """
    Achitecture ResNet
    """

    def __init__(self, num_classes=21, in_channels=1, depth=4, option="small", size=256, use_arcface=True):
        """
        initialisation des couches :
        num_classes = nombre de classe en sortie
        in_channels = nombre de representation de l'image passee en argument
        depth = 3, 4 ou 5 selon la taille des images en entrees
        option = small, medium or large
        size = taille des images pour le calcul de la couche dense apres le flatten
        """
        super(ResNet, self).__init__()
        self.size=size
        self.depth=depth
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes

        size=64

        #gestion de l'hyper paramêtre du nombre de couches et de la profondeur
        self.layer_size=calcul_layer_size(depth,option)
        t=len(self.layer_size)

        #première couche 
        self.first_layer=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.maxpool=nn.MaxPool2d(kernel_size=2,padding=0,stride=2)

        #descente en enfer
        self.down_layer=nn.ModuleList()

        for j in range(0,depth):
            # couches de bottleneck par étage:
            for i in range(0,self.layer_size[j]):
                self.down_layer.append(
                    nn.Sequential(
                        # 256 -> 64 -> 256
                        nn.Conv2d(in_channels=size*2**(j+2), out_channels=size*2**j, kernel_size=1, stride=1, padding="same"),
                        nn.BatchNorm2d(size*2**j),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=size*2**j, out_channels=size*2**j, kernel_size=3, stride=1, padding="same"),
                        nn.BatchNorm2d(size*2**j),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=size*2**j, out_channels=size*2**(j+2), kernel_size=1, stride=1, padding="same"),
                        nn.BatchNorm2d(size*2**(j+2)),
                    )
                )

            if(j<depth-1):
                # couche de convolution avec une stride de 2 pour la descente:   
                self.down_layer.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=size*2**(j+2), out_channels=size*2**(j+1), kernel_size=2, stride=2),
                        nn.BatchNorm2d(size*2**(j+1)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=size*2**(j+1), out_channels=size*2**(j+3), kernel_size=1, stride=1, padding="same"),
                        nn.BatchNorm2d(size*2**(j+3)),
                        nn.ReLU(inplace=True),
                    )
                )

        self.avgpool=nn.AvgPool2d(kernel_size=2, padding=0)

        #in_features=size*size*2**(len(depth)+2)
        #self.fc = nn.Linear(in_features=int(self.size/(2**(1+depth)))**2*2**(7+depth), out_features=num_classes)

        # ArcFace Hyperparameters
        arcFace_config = {
            "s": 30.0,  # scale (The scale parameter changes the shape of the logits. The higher the scale, the more peaky the logits vector becomes.)
            "m": 0.50,  # margin (margin results in a bigger separation of classes in your training set)
            "ls_eps": 0.0,
            "easy_margin": False
        }
        if use_arcface:
            self.fc = ArcMarginProduct(in_features=int(self.size/(2**(1+depth)))**2*2**(7+depth), 
                                   out_features=num_classes,
                                   s=arcFace_config["s"], 
                                   m=arcFace_config["m"], 
                                   easy_margin=arcFace_config["ls_eps"], 
                                   ls_eps=arcFace_config["ls_eps"])
        else:
            self.fc = nn.Linear(in_features=int(self.size/(2**(1+depth)))**2*2**(7+depth), out_features=num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu=nn.ReLU(inplace=True)


    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        bottleneck_output=0
        out=self.first_layer(x)
        out=self.maxpool(out)

        down_index=[sum(self.layer_size[0:i])+i-1 for i in range(1,len(self.layer_size))]
        count=0

        for layer in self.down_layer:
            if(count in down_index):
                #la couche qui sort est une couche de convolution stride 2 pour descendre
                out=layer(out)
                bottleneck_output=out
            else:
                #la couche qui sort est un bottleneck donc on effectue une somme avec la sortie précédente puis un relu
                out=layer(out)
                out+=bottleneck_output
                out=self.relu(out)
                bottleneck_output=out

            count+=1
        
        out = self.avgpool(out)
        out=out.flatten(start_dim=1)
        out = self.fc(out)
        out=self.softmax(out)
        return out

    
#calcul à partir des options la largeur de toutes les couches
def calcul_layer_size(depth,option):
    if (depth==4): # correspond au papier de recherche: https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8
        if (option=="small"):
            return [3,4,6,3]
        elif (option=="medium"):
            return [3,4,23,3]
        elif (option=="large"):
            return [3,8,36,3]
        else :
            print("mauvaise option: choisir entre small, medium et large ")
            print("execution du code avec l'option par defaut: small")
            return [3,4,6,3]
    elif (depth==3): # nos idees pour proposer plus d'option selon la taille des images par exemple
        if (option=="small"):
            return [3,4,3]
        elif (option=="medium"):
            return [3,6,3]
        elif (option=="large"):
            return [3,6,5]    
        else :
            print("mauvaise option: choisir entre small, medium et large ")
            print("execution du code avec l'option par defaut: small")
            return [3,4,3]
    elif (depth==5): # nos idees pour proposer plus d'option selon la taille des images par exemple
        if (option=="small"):
            return [3,4,6,4,3]
        elif (option=="medium"):
            return [3,4,23,6,3]
        elif (option=="large"):
            return [3,8,36,10,3]   
        else :
            print("mauvaise option: choisir entre small, medium et large ")
            print("execution du code avec l'option par defaut: small")
            return [3,4,6,4,3]
    else :
        print("mauvaise profondeur: choisir entre 3, 4 et 5 ")
        print("execution du code avec l'option par defaut: profondeur=4, option=small")
        return [3,4,6,3]
