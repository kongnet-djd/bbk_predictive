import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
from torchvision.models import densenet121, densenet201, resnet50, resnet101, vgg16, resnet18
import numpy as np
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from .backbone import resnet18

class net_1D(nn.Module):
    def __init__(self, num_mtdt, out_channel):
        super(net_1D, self).__init__()
        self.features = nn.Sequential(
                                     nn.Linear(num_mtdt, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, out_channel),
                                     nn.ReLU()
                                     )

    def forward(self, x):
        x = self.features(x)
        return x

class net_cmr(nn.Module):
    """
    Model for CMR
    """
    def __init__(self, out_channel):

        super(net_cmr, self).__init__()

        self.network_3D = resnet50(weights = None)
        # self.network_3D = resnet18(weights=None)

        # New input - 3D volume
        # self.network_3D.conv1 = torch.nn.Conv2d(img_size[2], 64, kernel_size=3, padding=1)
        self.network_3D.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)

        number_ftrs = self.network_3D.fc.in_features

        # Classifier
        self.network_3D.fc = nn.Sequential(
                                      nn.Linear(number_ftrs, out_channel),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        x = self.network_3D(x)
        return x

class net_cmr_mtdt(nn.Module):
    """
    Joint-Net
    """

    def __init__(self, args):
        super(net_cmr_mtdt, self).__init__()

        self.cmr_model = net_cmr(out_channel = 2048)
        self.mtdt_model = net_1D(num_mtdt=args.num_mtdt, out_channel=128)


        # Combination of features of all branches (i.e. Fundus and mtdt)
        self.combine = nn.Sequential(
                                    nn.Linear(128 + 2048, 2048),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(2048, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024, args.n_classes),
                                    # nn.Sigmoid(),
                                    )


    def forward(self, cmr, mtdt):


        u2 = self.cmr_model(cmr)
        v2 = self.mtdt_model(mtdt)

        # Combining all features from models
        concat_feats = torch.cat((u2,v2), 1)

        # After combining those features, they are later passed through a classifier
        combine = self.combine(concat_feats)

        return combine


# 下面这两个网络真的使垃圾，怎么都不行 OMG-GE垃圾

# class net_mtdt_omg_ge(nn.Module):
#     def __init__(self, args):
#         super(net_mtdt_omg_ge, self).__init__()
#
#         fusion = args.fusion_method
#         n_classes = 1
#         if fusion == 'sum':
#             self.fusion_module = SumFusion(output_dim=n_classes)
#         elif fusion == 'concat':
#             self.fusion_module = ConcatFusion(output_dim=n_classes)
#         elif fusion == 'film':
#             self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
#         elif fusion == 'gated':
#             self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
#         else:
#             raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
#
#
#         self.fundus_net = resnet18(modality='Retinal_fundus')
#         self.mtdt_model = net_1D(num_mtdt=args.num_mtdt, out_channel=512)
#
#     def forward(self, m1, m2):
#
#         a = self.fundus_net(m1) # [b, 512, 16,16]
#         v = self.mtdt_model(m2) # [b, 512]
#
#
#         # (_, C, H, W) = v.size()
#         # B = a.size()[0]
#         # v = v.view(B, -1, C, H, W)
#         # v = v.permute(0, 2, 1, 3, 4)
#
#         a = F.adaptive_avg_pool2d(a, 1) # [b, 512, 1,1]
#         # v = F.adaptive_avg_pool3d(v, 1)
#
#         a = torch.flatten(a, 1) # [b, 512]
#
#         # v = torch.flatten(v, 1)
#
#         out = self.fusion_module(a, v)
#
#         return out
#
# class net_cmr_mtdt_omgge(nn.Module):
#     def __init__(self, args):
#         super(net_cmr_mtdt_omgge, self).__init__()
#
#         self.cmr_model = net_cmr(out_channel=512)
#         self.mtdt_model = net_1D(num_mtdt=args.num_mtdt, out_channel=512)
#
#         self.fusion_module = ConcatFusion(output_dim=1)
#
#     def forward(self, cmr, mtdt):
#
#         u2 = self.cmr_model(cmr)
#         v2 = self.mtdt_model(mtdt)
#
#         f, o, out = self.fusion_module(u2,v2) # [batch_size , 512],[batch_size , 512],[batch_size , 1]
#
#         return f, o, out