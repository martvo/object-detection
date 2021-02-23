import torch
import torch.nn as nn
from torchvision.models import vgg19_bn, resnet34, densenet161

from typing import Any, Callable, Optional, Tuple, List


class YOLOv1(nn.Module):
    def __init__(self, 
                 feature_extractor: Callable, 
                 num_grid: int = 7, 
                 num_bboxes: int = 2, 
                 num_classes: int = 80) -> None:
        super(YOLOv1, self).__init__()
        """ Constructor
        Args:
            feature_extractor (Callable): network to use as a feature extractor
            num_grid (int, optional): [description]. Defaults to 7.
            num_bboxes (int, optional): [description]. Defaults to 2.
            num_classes (int, optional): [description]. Defaults to 80.
        """
        self.S = num_grid
        self.B = num_bboxes
        self.C = num_classes
        
        self.feature_extractor = feature_extractor
        self.conv_layers = self.create_conv_layers()
        self.fc_layers = self.create_fc_layers()
        
        # TODO: initialize weights manualy?

    def create_conv_layers(self) -> Callable:
        """ Create convolutional layers
        Returns:
            Callable: Sequence of 4 conv and leaky ReLU layers
        """
        # We can assume that the input tensor to the first conv layer will look like [N, 512, 14, 14]
        # TODO: add batch_norm?
        conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        return conv
        
    def create_fc_layers(self) -> Callable:
        """ Create linear layers
        Returns:
            Callable: Sequence of linear layers
        """
        # We can assume that the input tensor to the first fc layer will look like [N, 512, 7, 7]
        fc = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(4096, self.S * self.S * (5 * self.B + self.C)),
            nn.Sigmoid()
        )
        return fc
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Function used when calling this class
        Args:
            x (torch.Tensor): input to the neural network
        Returns:
            torch.Tensor: output of the neural network
        """
        x = self.feature_extractor(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        x = x.view(-1, self.S, self.S, 5 * self.B + self.C)
        return x


if __name__ == "__main__":
    feature_extractor = vgg19_bn(pretrained=True).features
    model = YOLOv1(feature_extractor)

    x = torch.rand(10, 3, 448, 448)
    y = model(x)
    print(y.shape)