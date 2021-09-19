# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def test_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
	Acc_array = 0.
	Prec_array = 0.
	Spe_array = 0.
	Rec_array = 0.
	IoU_array = 0.
	Dice_array = 0.
	HD_array = 0.
	sample_num = 0.
	result_list = []
	CEloss_list = []
	JAloss_list = []
	Label_list = []
	net.eval()
	result_list = []
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(dataloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).cuda()
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				inputs_batched = inputs_batched.cuda()
				# feature, seg_list, predicts = net(inputs_batched)
				feature, seg_list, predicts = net(inputs_batched)
				feature_batched = feature.clone()
				predicts_batched = predicts.clone()
				seg_batched1 = seg_list[0].clone()
				seg_batched2 = seg_list[1].clone()
				del feature
				del predicts
				del seg_list
				# del seg2			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			seg_batched1 = F.interpolate(seg_batched1, size=None, scale_factor=4, mode='bilinear', align_corners=True)
			seg_batched2 = F.interpolate(seg_batched2, size=None, scale_factor=4, mode='bilinear', align_corners=True)
			result_seg1 = torch.argmax(seg_batched1, dim=1).cpu().numpy().astype(np.uint8)
			result_seg2 = torch.argmax(seg_batched2, dim=1).cpu().numpy().astype(np.uint8)
			
			for i in range(batch):
				f = feature_batched[i,:,:,:].cpu().numpy()
				l = cv2.resize(labels_batched[i,:,:], dsize=(f.shape[1],f.shape[2]), interpolation=cv2.INTER_NEAREST)
				# np.save('/home/xiaoqiguo2/SemiSeg/vis/woL2uDT_labeled/'+name_batched[i].split('.')[0]+'feature.npy', f)
				# np.save('/home/xiaoqiguo2/SemiSeg/vis/woL2uDT_labeled/'+name_batched[i].split('.')[0]+'label.npy', l)
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				# p_seg1 = result_seg1[i,:,:]	
				# p_seg1 = cv2.resize(p_seg1, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				# cv2.imwrite('/home/xiaoqiguo2/SemiSeg/vis/SGM_group/seg'+name_batched[i], p_seg1*4)				
				# p_seg2 = result_seg2[i,:,:]					
				# p_seg2 = cv2.resize(p_seg2, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				# cv2.imwrite('/home/xiaoqiguo2/SemiSeg/vis/SGM_group/seg_'+name_batched[i], p_seg2*4)				
				l = labels_batched[i,:,:]
				#p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				#l = cv2.resize(l, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				predict = np.int32(p)
				gt = np.int32(l)
				cal = gt<255
				mask = (predict==gt) * cal 
				TP = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				TN = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				P = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				T = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)  

				P = np.sum((predict==1)).astype(np.float64)
				T = np.sum((gt==1)).astype(np.float64)
				TP = np.sum((gt==1)*(predict==1)).astype(np.float64)
				TN = np.sum((gt==0)*(predict==0)).astype(np.float64)

				Acc = (TP+TN)/(T+P-TP+TN)
				Prec = TP/(P+10e-6)
				Spe = TN/(P-TP+TN)
				Rec = TP/T
				DICE = 2*TP/(T+P)
				IoU = TP/(T+P-TP)
			#	HD = max(directed_hausdorff(predict, gt)[0], directed_hausdorff(predict, gt)[0])
				beta = 2
				HD = Rec*Prec*(1+beta**2)/(Rec+beta**2*Prec+1e-10)
				Acc_array += Acc
				Prec_array += Prec
				Spe_array += Spe
				Rec_array += Rec
				Dice_array += DICE
				IoU_array += IoU
				HD_array += HD
				sample_num += 1
				#p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				result_list.append({'predict':np.uint8(p*255), 'label':np.uint8(l*255), 'IoU':IoU, 'name':name_batched[i]})

		Acc_score = Acc_array*100/sample_num
		Prec_score = Prec_array*100/sample_num
		Spe_score = Spe_array*100/sample_num
		Rec_score = Rec_array*100/sample_num
		Dice_score = Dice_array*100/sample_num
		IoUP = IoU_array*100/sample_num
		HD_score = HD_array*100/sample_num
		print('%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n'%('Acc',Acc_score,'Sen',Rec_score,'Spe',Spe_score,'Prec',Prec_score,'Dice',Dice_score,'Jac',IoUP,'F2',HD_score))
	dataset.save_result(result_list, cfg.MODEL_NAME)

if __name__ == '__main__':
	test_net()


