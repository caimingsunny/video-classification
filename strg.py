from collections import OrderedDict
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from rgcn_models import RGCN

from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)

from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet

# 构建整个视频动作识别模型
class STRG(nn.Module):
    def __init__(self, base_model, in_channel=2048, out_channel=512,
                 nclass=174, dropout=0.3, nrois=10,
                 freeze_bn=True, freeze_bn_affine=True,
                 roi_size=7
                 ):
        super(STRG,self).__init__()
        self.base_model = base_model # 基础模型，用于提取特征
        self.in_channel = in_channel # 输入特征的通道数
        self.out_channel = out_channel # 输出特征的通道数
        self.nclass = nclass # 分类数
        self.nrois = nrois # 感兴趣区域的个数

        self.freeze_bn = freeze_bn # 是否冻结BatchNorm层
        self.freeze_bn_affine = freeze_bn_affine # 是否冻结BatchNorm层的weight和bias

        # 将基础模型的最后一层全连接层和平均池化层替换为Identity函数
        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.Identity()
        if False:
            self.base_model.maxpool.stride = (1,2,2)
            self.base_model.layer3[0].conv2.stride=(1,2,2)
            self.base_model.layer3[0].downsample[0].stride=(1,2,2)
            self.base_model.layer4[0].conv2.stride=(1,1,1)
            self.base_model.layer4[0].downsample[0].stride=(1,1,1)

        self.reducer = nn.Conv3d(self.in_channel, self.out_channel,1) # 将特征维度从 in_channel 到 out_channel
        self.classifier = nn.Linear(2*self.out_channel, nclass) # 二分类器 
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), # 自适应平均池化层
            nn.Dropout(p=dropout)
        )
        self.max_pool = nn.AdaptiveAvgPool2d(1) # 最大池化层

        self.strg_gcn = RGCN() # RGCN模型，用于处理感兴趣区域的特征
        self.roi_align = RoIAlign((roi_size,roi_size), 1/8, -1, aligned=True) # 将给定的 ROIs 对应到 CNN 的特征图上

    def extract_feature(self, x):
        return self.base_model.extract_feature(x)
        # 返回 base_model 的特征提取器提取的特征

#        x = self.base_model.conv1(x)
#        x = self.base_model.bn1(x)
#        x = self.base_model.relu(x)
#        if not self.base_model.no_max_pool:
#            x = self.base_model.maxpool(x)

#        x = self.base_model.layer1(x)
#        x = self.base_model.layer2(x)
#        x = self.base_model.layer3(x)
#        x = self.base_model.layer4(x)
#        return x


    def forward(self, inputs, rois=None):
        features = self.extract_feature(inputs)
        features = self.reducer(features) # 降维 N*C*T*H*W
        pooled_features = self.avg_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)
        # 对特征进行平均池化，并将多余的维度进行挤压，得到N*C的形式
        N, C, T, H, W = features.shape

        # 将ROIs张量重新调整形状，以适应后续处理
        rois_list = rois.view(-1, self.nrois, 4)
        rois_list = [r for r in rois_list]

        # 将特征张量进行转置和重新调整形状
        features = features.transpose(1,2).contiguous().view(N*T,C,H,W)
        
        # 将ROIs和特征张量进行ROI对齐和最大池化，然后重新调整形状
        rois_features = self.roi_align(features, rois_list)
        rois_features = self.max_pool(rois_features)
        rois_features = rois_features.view(N,T,self.nrois,C)
        
        # 将ROIs和特征张量进行ROI对齐和最大池化，然后重新调整形状
        gcn_features = self.strg_gcn(rois_features, rois)

        # 对调整后的特征进行空间时序GCN操作
        features = torch.cat((pooled_features, gcn_features), dim=-1)
        outputs = self.classifier(features)

        return outputs


    def train(self, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """

        super(STRG, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        # 如果冻结BatchNorm2D和权重，将BN层设置为评估模式，并设置不进行反向传播
        if self.freeze_bn:
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


if __name__ == '__main__':

    model = resnet.generate_model(model_depth=50,
                                    n_classes=174,
                                    n_input_channels=3,
                                    shortcut_type='B',
                                    conv1_t_size=7,
                                    conv1_t_stride=1,
                                    no_max_pool=False,
                                    widen_factor=1.0)

    rois = torch.rand((4,8,10,4))
    inputs = torch.rand((4,3,16,224,224))
    strg = STRG(model)
    out = strg(inputs, rois)

    pdb.set_trace()
    print(out.shape)
