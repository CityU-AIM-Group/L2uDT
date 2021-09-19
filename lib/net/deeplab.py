# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP

class refine_block(nn.Module):
    def __init__(self, c, k):
        super(refine_block, self).__init__()
        self.conv1 = nn.Conv2d(c, k, 1)#, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))  

    def forward(self, x):
        temp = 0.5 #1. #0.5
        idn = x
        _, c, _, _ = x.size()
        fc_out = self.conv1(x)
        seg = nn.Softmax(dim=1)(fc_out * temp)

        b, k, h, w = seg.size()
        seg_ = seg.view(b, k, h*w)                                       # b * k * n
        weights = torch.squeeze(self.conv1.weight).repeat(b, 1, 1)      # b * k * c
        weights = weights.permute(0, 2, 1)                              # b * c * k
        refine_feat = torch.bmm(weights, seg_)                           # b * c * n
        ref_x = refine_feat.view(b, c, h, w)                                # b * c * h * w

        ref_x = F.relu(ref_x, inplace=True)
        # The second 1x1 conv
        ref_x = self.conv2(ref_x)
        ref_x = 0.5*(ref_x + idn)
        refined_features = F.relu(ref_x, inplace=True)
        return refined_features, fc_out

class deeplab(nn.Module):
	def __init__(self, cfg):
		super(deeplab, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=16//cfg.MODEL_OUTPUT_STRIDE,
				bn_mom = cfg.TRAIN_BN_MOM)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				nn.BatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				nn.BatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				nn.BatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.RM0 = refine_block(cfg.MODEL_ASPP_OUTDIM, cfg.Cluster0)
		self.RM1 = refine_block(cfg.MODEL_ASPP_OUTDIM, cfg.Cluster1)
		# self.Matrix0 = nn.Linear(cfg.MODEL_NUM_CLASSES, cfg.Cluster0, bias=False)
		# self.Matrix1 = nn.Linear(cfg.Cluster0, cfg.Cluster1, bias=False)
		# self.RM2 = refine_block(cfg.MODEL_ASPP_OUTDIM, cfg.Cluster2)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()

	# def forward(self, x):
	# 	x_bottom = self.backbone(x)
	# 	layers = self.backbone.get_layers()
	# 	feature_aspp = self.aspp(layers[-1])
	# 	feature_aspp = self.dropout1(feature_aspp)
	# 	feature_aspp = self.upsample_sub(feature_aspp)

	# 	feature_shallow = self.shortcut_conv(layers[0])
	# 	feature_cat = torch.cat([feature_aspp,feature_shallow],1)
	# 	feature = self.cat_conv(feature_cat) 
	# 	result0, cluster0 = self.RM0(feature)
	# 	result1, cluster1 = self.RM1(feature)
	# 	# result2, cluster2 = self.RM2(feature)
	# 	feature1 = (result0+result1)/2.
	# 	result = self.cls_conv(feature)
	# 	result = self.upsample4(result)

	# 	return feature, [cluster0, cluster1], result

	def forward(self, x, feature_cat=None, phase='orig'):
		if phase  == 'orig':
			x_bottom = self.backbone(x)
			layers = self.backbone.get_layers()
			feature_aspp = self.aspp(layers[-1])
			feature_aspp = self.dropout1(feature_aspp)
			feature_aspp = self.upsample_sub(feature_aspp)

			feature_shallow = self.shortcut_conv(layers[0])
			feature_cat = torch.cat([feature_aspp,feature_shallow],1)

		feature = self.cat_conv(feature_cat) 
		result0, cluster0 = self.RM0(feature)
		result1, cluster1 = self.RM1(feature)
		# result2, cluster2 = self.RM2(feature)
		feature1 = (result0+result1)/2.
		result = self.cls_conv(feature)
		result = self.upsample4(result)

		return feature_cat, [cluster0, cluster1], result

