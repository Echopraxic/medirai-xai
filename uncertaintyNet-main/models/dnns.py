import torch.nn as nn
import torchvision.models as models

class ResNet50Hidden(nn.Module):
    def __init__(self, num_classes:int, weights="IMAGENET1K_V2"):
        super().__init__()
        self.model = models.resnet50(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 256),
            nn.ReLU6(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class ResNet18Hidden(nn.Module):
    def __init__(self, num_classes:int, weights="IMAGENET1K_V2"):
        super().__init__()
        self.model = models.resnet18(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 256),
            nn.ReLU6(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class WideResNet50Hidden(nn.Module):
    def __init__(self, num_classes:int, weights="IMAGENET1K_V2"):
        super().__init__()
        self.model = models.wide_resnet50_2(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 256),
            nn.ReLU6(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    