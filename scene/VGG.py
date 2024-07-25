import torch.nn as nn
from collections import namedtuple
import torchvision.models as models


import torchvision
normalize_vgg = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225])

def denormalize_vgg(img):
    im = img.clone()
    im[:, 0, :, :] *= 0.229
    im[:, 1, :, :] *= 0.224
    im[:, 2, :, :] *= 0.225
    im[:, 0, :, :] += 0.485
    im[:, 1, :, :] += 0.456
    im[:, 2, :, :] += 0.406
    return im
class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels,not_act=False):
        super(OneConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if not_act:
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x)
# pytorch pretrained vgg
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #pretrained vgg19
        vgg19 = models.vgg19(weights='DEFAULT').features

        self.relu1_1 = vgg19[:2]
        self.relu2_1 = vgg19[2:7]
        self.relu3_1 = vgg19[7:12]
        self.relu4_1 = vgg19[12:21]
        self.avgpool = nn.AdaptiveAvgPool2d(64)
        #fix parameters
        self.requires_grad_(False)
        for param in self.relu1_1.parameters():
            param.requires_grad = False
        for param in self.relu2_1.parameters():
            param.requires_grad = False
        for param in self.relu3_1.parameters():
            param.requires_grad = False
        for param in self.relu4_1.parameters():
            param.requires_grad = False
        backbone_channels= [ 64, 64, 128, 256, 512]
        self.mask_decoder=nn.Sequential(
                nn.ConvTranspose2d(backbone_channels[-1], backbone_channels[-2], kernel_size=2, stride=2),
                OneConv(backbone_channels[-2],backbone_channels[-2]),
                nn.ConvTranspose2d(backbone_channels[-2], backbone_channels[-3], kernel_size=2, stride=2),
                OneConv(backbone_channels[-3],backbone_channels[-3]),
                nn.ConvTranspose2d(backbone_channels[-3], backbone_channels[-4], kernel_size=2, stride=2),
                OneConv(backbone_channels[-4],backbone_channels[-4]),
                nn.Conv2d(backbone_channels[-4], 1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        _output = namedtuple('output', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'mask'])
        relu1_1 = self.relu1_1(x)
        relu2_1 = self.relu2_1(relu1_1)
        relu3_1 = self.relu3_1(relu2_1)
        relu3_1_avg=self.avgpool(relu3_1)
        relu4_1 = self.relu4_1(relu3_1)
        mask=self.mask_decoder(relu4_1)
        output = _output(normalize_(relu1_1), normalize_(relu2_1), normalize_(relu3_1_avg), normalize_(relu4_1), mask)
        return output
        # return output.relu3_1
        
def normalize_(a):
    return (a - a.mean()) / (a.max()-a.min())