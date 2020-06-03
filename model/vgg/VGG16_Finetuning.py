import torch 
import torch.nn as nn
from torchvision import models


class VGG16_Finetuning(nn.Module):

    def __init__(self, num_class):
        super(VGG16_Finetuning, self).__init__()
        self.net = models.vgg16(pretrained=True)
        
        # self.feature = net.features()
        # self.avgpool = net.avgpool()
        self.net.classifier[6] = nn.Linear(4096, num_class)


    def forward(self, x):
        x = self.net(x)
        return x