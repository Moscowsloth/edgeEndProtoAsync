import torch
import torch.nn.functional as F
from torch import nn

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

    # FedGKT的修改：返回提取的特征（客户端）
    def forward_features(self, x):
        out = self.conv1(x)
        features = self.conv2(out)
        out = torch.flatten(features, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out, features
    def forward_features_server(self, x):
        out = torch.flatten(x, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
    # Fedagg的修改
    def fedagg_forward_features_server(self, x):
        out = self.conv1(x)
        features = self.conv2(out)
        out = torch.flatten(features, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out, features

# Fedagg对比方法
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature1=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#64*16*16
        
        self.feature2=nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),#64*2*2
        nn.Linear(64,48),
        nn.ReLU(inplace=True),
        )
        
        self.fc=nn.Sequential(
        nn.Linear(48,10)
        )
    def forward(self, x):
        x = self.feature1(x)
        feature=x
        x=self.feature2(x)
        x = self.fc(x)
        return x, feature