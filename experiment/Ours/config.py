# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
		self.DATA_CROSS = ''
		self.EXP_NAME = 'CVC_L2uDT'+self.DATA_CROSS
		self.Cluster0 = 32
		self.Cluster1 = 64
		self.Cluster2 = 128
		self.gpu_id = '7'
		# os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

		self.Gen_advScale_eps = 0.05
		self.Gen_advScale_min = 0.1
		self.Gen_advScale_delta = 0.05
		self.Gen_advScale_lr = 10.
		self.Gen_advScale_iter = 10
		self.CL_weight = 0.05
		self.CL_weight_decay = 0.1
		self.CL_temp1 = None
		self.CL_temp2 = 0.8
		self.RMloss_temp = 0.5 #
		self.CL_Region_temp = 0.5 #
		self.L2u_temp = 1. #
		# self.is_shuffle = True

		self.DATA_NAME = 'CVC'
		self.DATA_AUG = False
		self.DATA_WORKERS = 2
		self.DATA_RESCALE = 256
		self.DATA_RANDOMCROP = 384
		self.DATA_RANDOMROTATION = 180
		self.DATA_RANDOMSCALE = 1.25
		self.DATA_RANDOM_H = 10
		self.DATA_RANDOM_S = 10
		self.DATA_RANDOM_V = 10
		self.DATA_RANDOMFLIP = 0.5
			
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res101_atrous'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 2
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)

		self.TRAIN_LR = 0.001
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.00004
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9 #0.9
		self.TRAIN_GPUS = 1
		self.TRAIN_BATCHES = 8
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0	
		self.TRAIN_EPOCHS = 500
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_TBLOG = True
		self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'/home/xiaoqiguo2/L2uDT/model/deeplabv3plus_res101_atrous_VOC2012_epoch46_all.pth')
		# self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'/home/xiaoqiguo2/SemiSeg/model/EMA_l2u_train1/model-best-deeplabv3plus_res101_atrous_CVC_epoch97_jac76.479.pth')

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

		self.TEST_MULTISCALE = [1.0]#[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
		self.TEST_FLIP = False#True
		self.TEST_CKPT = os.path.join(self.ROOT_DIR,'./model/CVC_SoCL/model-best-deeplabv3plus_res101_atrous_CVC_epoch67_jac74.393.pth')
		self.TEST_GPUS = 1
		self.TEST_BATCHES = 32	

		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)

cfg = Configuration() 
print('Cluster0:', cfg.Cluster0)
print('Cluster1:', cfg.Cluster1)
print('Gen_advScale_eps:', cfg.Gen_advScale_eps)
print('Gen_advScale_min:', cfg.Gen_advScale_min)
print('Gen_advScale_delta:', cfg.Gen_advScale_delta)
print('Gen_advScale_lr:', cfg.Gen_advScale_lr)
print('Gen_advScale_iter:', cfg.Gen_advScale_iter)
print('CL_weight:', cfg.CL_weight)
print('CL_weight_decay:', cfg.CL_weight_decay)
print('CL_temp1:', cfg.CL_temp1)
print('CL_temp2:', cfg.CL_temp2)
print('RMloss_temp:', cfg.RMloss_temp)
print('CL_Region_temp:', cfg.CL_Region_temp)
print('L2u_temp:', cfg.L2u_temp)
# print('is_shuffle:', cfg.is_shuffle)