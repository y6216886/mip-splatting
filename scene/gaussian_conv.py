import torch
import torch_scatter
# from sklearn.neighbors import NearestNeighbors
# from fast_pytorch_kmeans import KMeans
import torch.nn as nn
import torch.nn.functional as F
from math import log2


#######renderer
from kornia.filters import filter2d
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)


    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)
    
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.in_feature = in_feature
        # self.out_feature = out_feature
        self._make_layer()
        

    def _make_layer(self):
        self.layer_1 = nn.Conv2d(self.in_feature, self.in_feature * 2, 1, 1, padding=0)
        self.layer_2 = nn.Conv2d(self.in_feature * 2, self.in_feature * 4, 1, 1, padding=0)
        self.blur_layer = Blur()
        self.actvn = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x:torch.Tensor):
        y = x.repeat(1, 4, 1, 1)
        out = self.actvn(self.layer_1(x))
        out = self.actvn(self.layer_2(out))
        
        out = out + y
        out = F.pixel_shuffle(out, 2)
        out = self.blur_layer(out)
        
        return out
    
##codes are from headnerf https://github.com/CrisHY1995/headnerf.git
class NeuralRenderer(nn.Module):  ##the decoder of CR-NeRF

    def __init__(
            self, bg_type = "white", feat_nc=128, out_dim=3, final_actvn=True, min_feat=32, 
            **kwargs):
        super().__init__()
        
        self.bg_type = bg_type
        
        self.final_actvn = final_actvn
        self.n_feat = feat_nc
        self.out_dim = out_dim
        # self.n_blocks = int(log2(img_size[0]/featmap_size[0]))
        self.min_feat = min_feat
        self._make_layer()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** 0), self.min_feat))]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (0 + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (0)), self.min_feat),
                       max(self.n_feat // (2 ** (0 + 1)), self.min_feat), 1, 1,  padding=0)
                ]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        self.blur_layer=Blur()

        
        
    def forward(self, x):
        if len(x)!=1:
            x=x[0]
        # feat=self.feat_layers[0](x)
        # # feat=self.blur_layer(feat)
        # feat = self.actvn(feat)
        rgb=self.feat_2_rgb_list[0](x)
        
        
        # for idx in range(self.n_blocks):
        #     hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
        #     net = self.actvn(hid)
            
        #     rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
        #     if idx < self.n_blocks - 1:
        #         rgb = self.rgb_upsample(rgb)
        
        if self.final_actvn:
            rgbs = torch.sigmoid(rgb)
        return rgbs
    
    
# class GaussianConv(torch.nn.Module):
#     def __init__(self, xyz, input_channel=256, layers_channel=[256, 128, 64, 32, 3], downsample_layer=[], upsample_layer=[], K=8):
#         super(GaussianConv, self, ).__init__()
#         assert len(downsample_layer) == len(upsample_layer) == 0 or \
#             (len(downsample_layer) == len(upsample_layer) and max(downsample_layer) < min(upsample_layer)) ,\
#             'downsample_layer and upsample_layer must be the same length and satisfy max(downsample_layer) < min(upsample_layer) or both are empty lists'
        
#         self.K = K
#         self.N = xyz.shape[0]
#         self.downsample_layer = downsample_layer
#         self.upsample_layer = upsample_layer

#         self.init_kmeans_knn(xyz, len(downsample_layer))
#         self.init_conv_params(input_channel, layers_channel)

#     @torch.no_grad()
#     def init_kmeans_knn(self, xyz, len_sample_layer):
#         self.knn_indices = []
#         self.kmeans_labels = []

#         # get original knn_indices
#         xyz_numpy = xyz.cpu().numpy()
#         nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
#         nn.fit(xyz_numpy)
#         _, knn_indices = nn.kneighbors(xyz_numpy) # [N, K]
#         self.knn_indices.append(knn_indices) 

#         last_N = self.N
#         last_xyz = xyz

#         for i in range(len_sample_layer):
#             print('Using KMeans to cluster point clouds in level', i)
#             kmeans = KMeans(n_clusters=last_N//self.K, mode='euclidean', verbose=1)
#             self.kmeans_labels.append(kmeans.fit_predict(last_xyz)) # [N]
#             down_centroids = torch_scatter.scatter(last_xyz, self.kmeans_labels[-1], dim=0, reduce='mean') # [cluster_num=N//5, D]

#             # get knn_indices for downsampled point clouds
#             nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
#             nn.fit(down_centroids.cpu().numpy())
#             _, knn_indices = nn.kneighbors(down_centroids.cpu().numpy())
#             self.knn_indices.append(knn_indices)

#             last_N = down_centroids.shape[0]
#             last_xyz = down_centroids

#     def init_conv_params(self, input_channel, layers_channel):
#         self.kernels = []
#         self.bias = []
#         for out_channel in layers_channel:
#             self.kernels.append(torch.randn(out_channel, self.K*input_channel)*0.1)  # [out_channel, K*input_channel]
#             self.bias.append(torch.zeros(1, out_channel))  # [1, out_channel]
#             input_channel = out_channel

#         self.kernels = torch.nn.ParameterList(self.kernels)
#         self.bias = torch.nn.ParameterList(self.bias)

#     def forward(self, features):
#         '''
#         Args:
#             features: [N, D]
#             D: input_channel
#             S: output_channel
#         '''
#         sample_level = 0
#         for i in range(len(self.kernels)):
#             if i in self.downsample_layer:
#                 sample_level += 1
#                 features = torch_scatter.scatter(features, self.kmeans_labels[sample_level-1], dim=0, reduce='mean')
#             elif i in self.upsample_layer:
#                 sample_level -= 1
#                 features = features[self.kmeans_labels[sample_level]]

#             knn_indices = self.knn_indices[sample_level]

#             knn_features = features[knn_indices] # [N, K, D]
#             knn_features = knn_features.reshape(knn_features.size(0), -1) # [N, K*D]
#             features = knn_features @ self.kernels[i].T + self.bias[i] # [N, S]
#             features = torch.sigmoid(features) if i != len(self.kernels)-1 else features

#         return features # [N, S]
    


class decoder3(nn.Module):
    def __init__(self,feat_nc=256, out_dim=3):
        super(decoder3,self).__init__()
        # decoder
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(feat_nc,128,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(128,128,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(128,64,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(64,64,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(64,out_dim,3,1,0)

    def forward(self,x):
        x=x[0]
        output = {}
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out_relu9 = self.relu9(out)
        out = self.unpool2(out_relu9)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out

if __name__ == '__main__':
    xyz = torch.randn(100, 3)
    features = torch.randn(1, 256, 256, 256).cuda()
    gaussian_conv = NeuralRenderer(feat_nc=256, out_dim=3).cuda()
    output = gaussian_conv(features)
    print(output.shape)