from __future__ import annotations
import torch
from torch import nn

class FastCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(1,24,5,2,2),nn.BatchNorm2d(24),nn.ReLU(inplace=True),
            nn.Conv2d(24,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
            nn.Conv2d(48,80,3,2,1),nn.BatchNorm2d(80),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.20),nn.Linear(80,out_dim))
    def forward(self,x): return self.h(self.f(x))

class DilatedCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(1,16,5,2,2),nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,2,2,dilation=2),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.20),nn.Linear(64,out_dim))
    def forward(self,x): return self.h(self.f(x))

class DepthwiseCNN(nn.Module):
    def __init__(self,out_dim):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(1,16,3,2,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,2,1,groups=16),nn.Conv2d(16,32,1),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,2,1,groups=32),nn.Conv2d(32,64,1),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.15),nn.Linear(64,out_dim))
    def forward(self,x): return self.h(self.f(x))

MODEL_FACTORIES={'fastcnn':FastCNN,'dilatedcnn':DilatedCNN,'depthwisecnn':DepthwiseCNN}
def build_model(name,out_dim): return MODEL_FACTORIES[name](out_dim)
