import torch
import torch.nn as nn
from utils.loss_utils import calc_mean_std

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()
        # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,matrixSize,3,1,1))
        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)


class MulLayer(nn.Module):
    def __init__(self, matrixSize=32, adain=True):
        super(MulLayer,self).__init__()
        self.adain = adain
        if adain:
            return

        self.snet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, matrixSize)
        )
        self.unzip = nn.Sequential(
            nn.Linear(matrixSize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )


    def forward(self,cF,sF, trans=True):
        '''
        input:
            point cloud features: [N, C]
            style image features: [1, C, H, W]
            D: matrixSize
        '''
        if self.adain:
            cF = cF.T # [C, N]
            style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
            content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

            style_mean = style_mean.squeeze(0)
            style_std = style_std.squeeze(0)
            content_mean = content_mean.squeeze(0)
            content_std = content_std.squeeze(0)

            cF = (cF - content_mean) / content_std
            cF = cF * style_std + style_mean
            return cF.T
      
        assert cF.size(1) == sF.size(1), 'cF and sF must have the same channel size'
        assert sF.size(0) == 1, 'sF must have batch size 1'
        N, C = cF.size()
        B, C, H, W = sF.size()

        # normalize point cloud features
        cF = cF.T # [C, N]
        style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
        content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

        content_mean = content_mean.squeeze(0)
        content_std = content_std.squeeze(0)

        cF = (cF - content_mean) / content_std # [C, N]
        # compress point cloud features
        compress_content = self.compress(cF.T).T # [D, N]

        # normalize style image features
        sF = sF.view(B,C,-1)
        sF = (sF - style_mean) / style_std  # [1, C, H*W]

        if(trans):
            # get style transformation matrix
            sMatrix = self.snet(sF.reshape(B,C,H,W)) # [B=1, D*D]
            sMatrix = sMatrix.view(self.matrixSize,self.matrixSize) # [D, D]

            transfeature = torch.mm(sMatrix, compress_content).T # [N, D]
            out = self.unzip(transfeature).T # [C, N]

            style_mean = style_mean.squeeze(0) # [C, 1]
            style_std = style_std.squeeze(0) # [C, 1]

            out = out * style_std + style_mean
            return out.T # [N, C]
        else:
            out = self.unzip(compress_content.T) # [N, C]
            out = out * content_std + content_mean
            return out




class CNNv1(nn.Module):
    def __init__(self,matrixSize=32,in_channel=64):
        super(CNNv1,self).__init__()
        # if(layer == 'r31'):
            # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(in_channel,128,1,1,0),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(128,64,1,1,0),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(64,matrixSize,1,1,0))
        # elif(layer == 'r41'):
        #     # 512x32x32
        #     self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(256,128,3,1,1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(128,matrixSize,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)


"""
The code below are from linear style transfer
""" 
class MulLayerv1(nn.Module):
    def __init__(self,matrixSize=32,in_channel=256):
        super(MulLayerv1,self).__init__()
        self.snet = CNNv1(matrixSize,in_channel=in_channel)
        self.cnet = CNNv1(matrixSize,in_channel=in_channel)
        self.matrixSize = matrixSize

        # if(layer == 'r41'):
        #     self.compress = nn.Conv2d(512,matrixSize,1,1,0)
        #     self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        # elif(layer == 'r31'):
        self.compress = nn.Conv2d(in_channel,matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,in_channel,1,1,0)
        self.transmatrix = None
        self.lifting = nn.Conv2d(256,32,1,1,0)

    def forward(self,cF,sF,trans=True):#两阶段的话，数值范围不匹配
        sF=self.lifting(sF)
        cFBK = cF.clone()
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS


        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return 