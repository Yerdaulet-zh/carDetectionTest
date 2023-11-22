import torch
import torch.nn as nn
import torch.nn.functional as F



def Conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.Mish(inplace=True))



class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        reduced_dim = in_channels // 2
        self.layer1 = Conv(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1, padding=0)
        self.layer2 = Conv(in_channels=reduced_dim, out_channels=in_channels, kernel_size=3, padding=1)
        
        
    def forward(self, x):
        x += self.layer2(self.layer1(x))
        return x
    


class Darknet(nn.Module):
    def __init__(self, in_channels, n_repeats=[1, 2, 8, 8, 4], num_classes=1000):
        super(Darknet, self).__init__()
        self.in_channels = 64
        self.extractor = nn.Sequential(
            Conv(in_channels, out_channels=32, kernel_size=3),
            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        )
        self.layer1 = self._make_layers(num_rep=n_repeats[0])
        self.layer2 = self._make_layers(num_rep=n_repeats[1])
        self.layer3 = self._make_layers(num_rep=n_repeats[2])
        self.layer4 = self._make_layers(num_rep=n_repeats[3])
        self.layer5 = self._make_layers(num_rep=n_repeats[4], transition=False)
#         self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
#         self.fc = nn.Linear(in_features=1024, out_features=num_classes)        
        
    
    def forward(self, x):
        x = self.extractor(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x4 = self.layer5(x3)
        # x = self.avg_pool(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        return x1, x2, x3, x4
        
        
    def _make_layers(self, num_rep, transition=True):
        layers = []
        
        for i in range(num_rep):
            layers.append(
                Residual(self.in_channels)
            )
        
        if transition:
            layers.append(
                Conv(self.in_channels, out_channels=self.in_channels*2, kernel_size=3, padding=1, stride=2)
            )
            self.in_channels *= 2
        return nn.Sequential(*layers)
    
    
    
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        
        self.L1 = Conv(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.L2 = Conv(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.L3 = Conv(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.L4 = Conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x1, x2, x3, x4):
        x4 = self.L1(x4)
        x3 = self.L2(x3) + x4
        x2 = self.L3(x2) + F.interpolate(x3, size=(14, 14))
        x1 = self.L4(x1) + F.interpolate(x2, size=(28, 28))
        return x1, x2, x3, x4

    
    
class PAN(nn.Module):
    def __init__(self, ):
        super(PAN, self).__init__()
        self.L1 = Conv(in_channels=512, out_channels=512, stride=2)
        self.L2 = Conv(in_channels=1024, out_channels=1024, stride=2)
        # self.L3 = Conv(in_channels=1024, out_channels=512, stride=2)
        
        
    def forward(self, x1, x2, x3, x4):
        x2 = torch.cat([x2, self.L1(x1)], dim=1)        
        x3 = torch.cat([x3, self.L2(x2)], dim=1)
        x4 = torch.cat([x4, x3], dim=1)
        return x1, x2, x3, x4
    
    
    
class FeatureMap(nn.Module):
    def __init__(self, in_channels):
        super(FeatureMap, self).__init__()
        self.darkmet53 = Darknet(in_channels)
        self.fpn = FPN()
        self.pan = PAN()
        
    def forward(self, x):
        x1, x2, x3, x4 = self.darkmet53(x)
        x1, x2, x3, x4 = self.fpn(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.pan(x1, x2, x3, x4)
        return x1, x2, x3, x4
    
    
    
class Architecture(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Architecture, self).__init__()
        self.featureMap = FeatureMap(in_channels)
        self.L1 = nn.Conv2d(in_channels=512, out_channels=num_classes + 1 + 4, kernel_size=1, stride=1, padding=0)
        self.L2 = nn.Conv2d(in_channels=1024, out_channels=num_classes + 1 + 4, kernel_size=1, stride=1, padding=0)
        self.L3 = nn.Conv2d(in_channels=1536, out_channels=num_classes + 1 + 4, kernel_size=1, stride=1, padding=0)
        self.L4 = nn.Conv2d(in_channels=2048, out_channels=num_classes + 1 + 4, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        x1, x2, x3, x4 = self.featureMap(x)
        x1 = self.L1(x1)
        x2 = self.L2(x2)
        x3 = self.L3(x3)
        x4 = self.L4(x4)
        return x1, x2, x3, x4
    
    
    
    
    
    
    
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        
        if self.downsample is not None:
            x += self.downsample(identity)
        return x
    
class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.LeakyReLU()
        
        self.layer1 = self._make_layer(BasicBlock, out_channels=64, num_rep=layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, out_channels=128, num_rep=layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, out_channels=256, num_rep=layers[2], stride=2)
        
        self.l1 = nn.Conv2d(in_channels=128, out_channels=(num_classes + 5) * 2, kernel_size=1, stride=1)
        self.l2 = nn.Conv2d(in_channels=256, out_channels=(num_classes + 5) * 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.maxpool(self.relu(x))
        
        x = self.layer1(x)
        x1 = self.layer2(x)
        x = self.layer3(x1)
        
        return self.l1(x1), self.l2(x)
    
    def _make_layer(self, BasicBlock, out_channels, num_rep, stride):
        layers = []
        
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if out_channels == 64:
            layers.append(BasicBlock(self.in_channels, out_channels))
        else:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=2, downsample=downsample))
        self.in_channels = out_channels
        
        for i in range(num_rep-1):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
# ResNet18 = ResNet(in_channels=3, num_classes=3, BasicBlock=BasicBlock, layers=[2, 2, 2])
# ResNet34 = ResNet(in_channels=3, num_classes=3, BasicBlock=BasicBlock, layers=[3, 4, 6])
