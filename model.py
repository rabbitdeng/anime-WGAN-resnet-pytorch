import torch
import torch.nn as nn
from config import opt
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

nz = 100
nc = 3
M = 32


class BasicBlock(nn.Module):
    def __init__(self, in1):  #in1为输入的channel大小，BasicBlock输出等于输入channel大小
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in1, in1 * 2, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 =nn.BatchNorm2d(in1*2)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in1*2, in1, kernel_size=3,
                        stride=1, padding=1, bias=False)
        self.bn2 =nn.BatchNorm2d(in1)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
      #  out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
      #  out = self.bn2(out)
        out = self.relu2(out)

        out = out + residual
        return out

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3 ,stride=1, padding=1)


        self.layer1 = nn.Sequential(
            BasicBlock(64),
            nn.AvgPool2d(3, 2),
            BasicBlock(64),
        )

        self.layer2 = nn.Sequential(
            nn.AvgPool2d(3,2),
            BasicBlock(64)
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool2d(3, 2),
            BasicBlock(64),
        )

        self.layer4 = nn.Sequential(
            nn.AvgPool2d(3, 2),
            BasicBlock(64),
        )

        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
          nn.Linear(576, 1),
        )
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


class netG(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(nz, 64*2*2)
        self.layer1 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2),
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2),

        )
        self.layer2 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.Conv = nn.Sequential(
            BasicBlock(64),
            #nn.BatchNorm2d(64),
            nn.LayerNorm([64,64,64]),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
            nn.Tanh()
        )


    def forward(self, z):
        x = self.linear(z)
        x = x.view(opt.batchsize,64,2,2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.Conv(x)
        return x

