# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import torch.nn.functional as F
import cv2
import math

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.loss import MaskCrossEntropyLoss, MaskBCELoss, MaskBCEWithLogitsLoss
from net.sync_batchnorm.replicate import patch_replication_callback
from scipy.spatial.distance import directed_hausdorff
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)
    
def Jaccard_loss(true, logits, eps=1e-7):
    intersection = torch.sum(logits * true, dim=(1,2))
    cardinality = torch.sum(logits + true, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1. - jacc_loss)

def EntropyLoss(inputs):
    log_likelihood = torch.log(inputs)
    loss = torch.mean(torch.mul(log_likelihood, inputs))
    return loss

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)

def make_one_hot(labels, class_num=2):
    target = torch.eye(class_num)[labels.long()]
    # gt_1_hot = target.permute(0, 3, 1, 2).float().cuda()
    return target.float().cuda()

def Cosine_dist(x, temp = 1):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    x_norm = x/(torch.norm(x, dim=1, keepdim=True) + 10e-10)
    dist = x_norm.mm(x_norm.permute(1, 0))
    return dist/temp

def L2U_F(feature, P0, P1, GroupMap0, GroupMap1, un_P0, un_P1):
    GroupMap0 = GroupMap0/(torch.sum(GroupMap0, dim=1, keepdim=True) + 10e-10)              # b * num_clusters * n
    GroupMap1 = GroupMap1/(torch.sum(GroupMap1, dim=1, keepdim=True) + 10e-10)
    b, c, h, w = feature.size()                            
    delta_P0 = un_P0 - P0              # b * num_clusters * c
    delta_P1 = un_P1 - P1
    delta_out0 = GroupMap0.permute(0, 2, 1).bmm(delta_P0)  ### b * n * c
    delta_out1 = GroupMap1.permute(0, 2, 1).bmm(delta_P1)  ### b * n * c
    delta_out0 = delta_out0.reshape(b, h, w, c).permute(0, 3, 1, 2)
    delta_out1 = delta_out1.reshape(b, h, w, c).permute(0, 3, 1, 2)
    return delta_out0, delta_out1

def Gen_advScale(feature, delta_out0, delta_out1, net, epoch):
    f = feature.clone().detach()
    delta0 = delta_out0.clone().detach()
    delta1 = delta_out1.clone().detach()

    eps = cfg.Gen_advScale_eps
    adv_distance = 1.
    delta_lambda = cfg.Gen_advScale_delta * (epoch // (cfg.Gen_advScale_delta*200) + 1)
    lambda_P = torch.ones(cfg.TRAIN_BATCHES, 1, 1, 1) 

    pred_nobias = net.cls_conv(f)
    pred_nobias = nn.UpsamplingBilinear2d(scale_factor=4)(pred_nobias)
    pred_nobias = torch.argmax(pred_nobias, dim=1).detach().float()
    i = 0
    while adv_distance > eps:
        lambda_P = to_var(lambda_P)
        feature_l2u = f + lambda_P * (delta0 + delta1)/2. 
        pred_l2u = net.cls_conv(feature_l2u)
        pred_l2u = nn.UpsamplingBilinear2d(scale_factor=4)(pred_l2u)

        adv_distance = Jaccard_loss(pred_nobias, torch.softmax(pred_l2u*50., dim=1)[:,1,:,:], eps=1e-7)
        net.zero_grad()
        adv_distance.backward()
        lambda_new = torch.clamp(lambda_P - cfg.Gen_advScale_lr*lambda_P.grad, min=0.0, max=1.0)
        lambda_P = lambda_new.clone().detach()
        del lambda_new
        i += 1
        if i >= cfg.Gen_advScale_iter:
            break
    lambda_P = torch.clamp(lambda_P, min=cfg.Gen_advScale_min, max=cfg.Gen_advScale_min+delta_lambda)
    # if epoch % 10 == 0:
    #     print(adv_distance, torch.mean(lambda_P))
    return lambda_P

def Gen_prototypes(feature, label, Cluster=None, num_clusters=2):
    b, c, h, w = feature.size()                            
    x_t = feature.permute(0, 2, 3, 1)              # b * h * w * c
    x_t = x_t.reshape(b, h*w, c)              # b * n * c

    label_4down = torch.squeeze(torch.nn.UpsamplingNearest2d(scale_factor=1/4)(torch.unsqueeze(label.float(),1)))
    GroupMap = make_one_hot(label_4down, class_num=cfg.MODEL_NUM_CLASSES)
    GroupMap = GroupMap.reshape(b, h*w, cfg.MODEL_NUM_CLASSES).permute(0, 2, 1)              # b * cfg.MODEL_NUM_CLASSES * n
    if num_clusters != cfg.MODEL_NUM_CLASSES:
        Group_n = Cluster.mul(torch.unsqueeze(torch.ones_like(label_4down)-label_4down, 1))[:,:num_clusters//2,:,:]
        Group_p = Cluster.mul(torch.unsqueeze(label_4down, 1))[:,num_clusters//2:,:,:]
        GroupMap = torch.cat([Group_n, Group_p], 1)
        GroupMap = GroupMap.permute(0, 2, 3, 1)
        GroupMap = GroupMap.reshape(b, h*w, num_clusters).permute(0, 2, 1)              # b * num_clusters * n

    ## Grouping maps & Prototypes
    GroupMap_area = torch.sum(GroupMap, dim=(2), keepdim=True)              # b * num_clusters * 1
    Prototypes = GroupMap.bmm(x_t)              # b * num_clusters * c
    Prototypes = Prototypes/(GroupMap_area + 10e-10)              # b * num_clusters * c
    return GroupMap, Prototypes

def Prototypes_CL(label, Cluster, prototype, Sigma=None):
    b, k, n = Cluster.size()  
    b, k, c = prototype.size()                           
    # Obtain Co-prototypes matrix
    prototype_all = prototype.reshape(b * k, c)
    # prototype = prototype.sum(0)

    Proto_Matrix = prototype_all.mm(prototype_all.permute(1, 0))
    # Proto_Matrix = Cosine_dist(prototype_all, temp = cfg.CL_temp1)
    Proto_Matrix = Proto_Matrix.reshape(b * k, b * k)

    # Assign labels for prototype according to the prototype_class
    label_4down = torch.squeeze(torch.nn.UpsamplingNearest2d(scale_factor=1/4)(torch.unsqueeze(label.float(),1)))
    label_4down = make_one_hot(label_4down, class_num=cfg.MODEL_NUM_CLASSES).reshape(b*n, cfg.MODEL_NUM_CLASSES)
    label_4down = torch.unsqueeze(label_4down, 1)           # (b * n) * 1 * cfg.MODEL_NUM_CLASSES
    group = torch.unsqueeze(Cluster.permute(0, 2, 1).reshape(b*n, k), 2)          # (b * n) * num_clusters * 1
    group = group.bmm(label_4down)          # (b * n) * num_clusters * cfg.MODEL_NUM_CLASSES
    group = group.reshape(b, n, k, cfg.MODEL_NUM_CLASSES)          # (b * n) * num_clusters * cfg.MODEL_NUM_CLASSES
    group = torch.sum(group, dim = (1))
    group = group.reshape(b * k, cfg.MODEL_NUM_CLASSES)          # (b * num_clusters) * cfg.MODEL_NUM_CLASSES
    prototype_zero = torch.squeeze(torch.nonzero(group[:, 0]))
    prototype_one = torch.squeeze(torch.nonzero(group[:, 1]))
 
    # Obtain prototype-based InfoNCE loss
    M_zero = torch.index_select(Proto_Matrix, 0, prototype_zero)
    M_one = torch.index_select(Proto_Matrix, 0, prototype_one)

    Proto_Matrix_sorted = torch.cat([M_zero, M_one], 0)
    dist_mat = Cosine_dist(Proto_Matrix_sorted, temp = cfg.CL_temp2)
    dist_mat = torch.exp(dist_mat)
    len0, _ = M_zero.size()
    len1, _ = M_one.size()

    zero_neg_sum = torch.sum(dist_mat[:len0,len0:], dim = (1), keepdim=True) # len0 * 1
    numerator = dist_mat[:len0,:len0] #len0 * len0
    zero_loss = 0. - torch.log(numerator / (numerator + zero_neg_sum + 10e-10))
    mask = torch.eye(len0).cuda()
    mask = torch.ones_like(mask) - mask
    zero_loss = (mask * zero_loss).mean()

    one_neg_sum = torch.sum(dist_mat[len0:,:len0], dim = (1), keepdim=True) # len0 * 1
    numerator = dist_mat[len0:,len0:] #len0 * len0
    one_loss = 0. - torch.log(numerator / (numerator + one_neg_sum + 10e-10))
    mask = torch.eye(len1).cuda()
    mask = torch.ones_like(mask) - mask
    one_loss = (mask * one_loss).mean()

    loss = zero_loss + one_loss
    return loss

def loss_one_batch(feature, GroupMap_list, pred, label, Sigma):
    ############
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    CE_Loss = criterion(pred, label)
    
    ############
    temp = cfg.RMloss_temp
    GroupMap0_n = nn.Softmax(dim=1)(GroupMap_list[0] * temp)[:,:cfg.Cluster0//2,:,:]
    GroupMap0_p = nn.Softmax(dim=1)(GroupMap_list[0] * temp)[:,cfg.Cluster0//2:,:,:]
    GroupMap1_n = nn.Softmax(dim=1)(GroupMap_list[1] * temp)[:,:cfg.Cluster1//2,:,:]
    GroupMap1_p = nn.Softmax(dim=1)(GroupMap_list[1] * temp)[:,cfg.Cluster1//2:,:,:]
    Entropy_loss = EntropyLoss(torch.sum(GroupMap0_n, dim=(0,2,3))/(GroupMap0_n.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap0_p, dim=(0,2,3))/(GroupMap0_p.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap1_n, dim=(0,2,3))/(GroupMap1_n.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap1_p, dim=(0,2,3))/(GroupMap1_p.sum()+10e-6))

    label_4down = torch.squeeze(torch.nn.UpsamplingNearest2d(scale_factor=1/4)(torch.unsqueeze(label.float(),1)))
    MP_loss = Jaccard_loss(label_4down, torch.sum(GroupMap0_p, dim=(1)), eps=1e-7)
    MP_loss += Jaccard_loss(label_4down, torch.sum(GroupMap1_p, dim=(1)), eps=1e-7)

    ############
    temp = cfg.CL_Region_temp
    GroupMap0 = nn.Softmax(dim=1)(GroupMap_list[0] * temp).clone().detach()
    GroupMap1 = nn.Softmax(dim=1)(GroupMap_list[1] * temp).clone().detach()        

    GroupMap0, P0 = Gen_prototypes(feature, label, Cluster=GroupMap0, num_clusters=cfg.Cluster0)
    GroupMap1, P1 = Gen_prototypes(feature, label, Cluster=GroupMap1, num_clusters=cfg.Cluster1)

    ProtoLoss0 = Prototypes_CL(label, GroupMap0, P0, Sigma=Sigma[:, :cfg.Cluster0, :])
    ProtoLoss1 = Prototypes_CL(label, GroupMap1, P1, Sigma=Sigma[:, cfg.Cluster0:, :])
    return Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1

def train_one_batch(inputs, labels, un_inputs, un_labels, net, optimizer, epoch):
    # eye = torch.eye(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM)
    # eye = torch.unsqueeze(eye, 0).repeat(cfg.Cluster0+cfg.Cluster1,1,1)
    eye = torch.ones(cfg.TRAIN_BATCHES, cfg.Cluster0+cfg.Cluster1, cfg.MODEL_ASPP_OUTDIM)
    Sigma = to_var(eye)

    ##### Actual train #####
    feature, GroupMap_list, pred = net(inputs)
    Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1 = loss_one_batch(feature, GroupMap_list, pred, labels, Sigma)
    loss = CE_Loss + MP_loss + Entropy_loss + cfg.CL_weight * (ProtoLoss0 + ProtoLoss1) * (cfg.CL_weight_decay ** int(epoch >= 50)) * (cfg.CL_weight_decay ** int(epoch >= 100))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ##### Data Augmentation or Semi-supervised #####
    feature, GroupMap_list, pred = net(inputs)
    un_feature, un_GroupMap_list, un_pred = net(un_inputs)
    pseudo_label = torch.from_numpy(torch.argmax(un_pred, dim=1).detach().cpu().numpy()).cuda()

    temp = cfg.L2u_temp
    GroupMap0 = nn.Softmax(dim=1)(GroupMap_list[0] * temp).clone().detach()
    GroupMap1 = nn.Softmax(dim=1)(GroupMap_list[1] * temp).clone().detach()
    un_GroupMap0 = nn.Softmax(dim=1)(un_GroupMap_list[0] * temp).clone().detach()
    un_GroupMap1 = nn.Softmax(dim=1)(un_GroupMap_list[1] * temp).clone().detach()    

    GroupMap0, P0 = Gen_prototypes(feature, labels, Cluster=GroupMap0, num_clusters=cfg.Cluster0)
    GroupMap1, P1 = Gen_prototypes(feature, labels, Cluster=GroupMap1, num_clusters=cfg.Cluster1)
    un_GroupMap0, un_P0 = Gen_prototypes(un_feature, pseudo_label, Cluster=un_GroupMap0, num_clusters=cfg.Cluster0)
    un_GroupMap1, un_P1 = Gen_prototypes(un_feature, pseudo_label, Cluster=un_GroupMap1, num_clusters=cfg.Cluster1)

    # print(torch.max(torch.abs(P0-un_P0)) + torch.max(torch.abs(P1-un_P1)))
    delta_out0, delta_out1 = L2U_F(feature, P0, P1, GroupMap0, GroupMap1, un_P0, un_P1)
    advScale = Gen_advScale(feature, delta_out0, delta_out1, net, epoch)
    # advScale = 0.9
    l2u_feature = feature + advScale * (delta_out0 + delta_out1)/2.    
    pred = net.cls_conv(l2u_feature)
    pred = nn.UpsamplingBilinear2d(scale_factor=4)(pred)

    Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1 = loss_one_batch(l2u_feature, GroupMap_list, pred, labels, Sigma)
    loss = CE_Loss #+ Entropy_loss + MP_loss #+ cfg.CL_weight * (ProtoLoss0 + ProtoLoss1) * (cfg.CL_weight_decay ** int(epoch >= 100)) * (cfg.CL_weight_decay ** int(epoch >= 150))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    del GroupMap0
    del GroupMap1
    del un_GroupMap0
    del un_GroupMap1

    return Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1

def train_net():
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'labeled', cfg.DATA_AUG)
    dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

    un_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'unlabeled', cfg.DATA_AUG)
    un_dataloader = DataLoader(un_dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

    test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
    test_dataloader = DataLoader(test_dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)

    net = generate_net(cfg)
    net.cuda()		

    print('Use %d GPU'%cfg.TRAIN_GPUS)
    device = torch.device(0)
    net = nn.DataParallel(net)
    patch_replication_callback(net)
    net.to(device)		

    if cfg.TRAIN_CKPT:
        pretrained_dict = torch.load(cfg.TRAIN_CKPT)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)

    # for i, para in enumerate(net.named_parameters()):
    #     (name, param) = para
    #     print(i, name)	
    # print(i, name)[0]	
    net = net.module

    segment_dict = []
    backbone_dict = []
    for i, para in enumerate(net.parameters()):
        if i < 48: 
            segment_dict.append(para)
        else:
            backbone_dict.append(para)

    optimizer = optim.SGD(
        params = [
            {'params': backbone_dict, 'lr': cfg.TRAIN_LR},
            {'params': segment_dict, 'lr': 10*cfg.TRAIN_LR}
        ],
        momentum=cfg.TRAIN_MOMENTUM)

    itr = cfg.TRAIN_MINEPOCH * len(dataloader)
    max_itr = cfg.TRAIN_EPOCHS * len(dataloader)
    best_jacc = 0.
    best_epoch = 0
    for epoch in range(cfg.TRAIN_MINEPOCH, 200):
        seg_entropy_running_loss = 0.0
        seg_mp_running_loss = 0.0
        seg_ce_running_loss = 0.0
        seg_proto_running_loss1 = 0.0
        seg_proto_running_loss2 = 0.0
        net.train()
        for i_batch, (sample_batched, un_batched) in enumerate(zip(dataloader, un_dataloader)):
            now_lr = adjust_lr(optimizer, itr, max_itr)
            inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
            labels_batched = labels_batched.long().cuda()
            inputs_batched = inputs_batched.cuda()
            inputs_un, labels_un = un_batched['image'], un_batched['segmentation']
            labels_un = labels_un.long().cuda()
            inputs_un = inputs_un.cuda()
                
            Entropy_loss, MP_loss, CE_Loss, ProtoLoss1, ProtoLoss2 = train_one_batch(inputs_batched, labels_batched, inputs_un, labels_un, net, optimizer, epoch)

            seg_entropy_running_loss += Entropy_loss.item()
            seg_mp_running_loss += MP_loss.item()
            seg_ce_running_loss += CE_Loss.item()
            seg_proto_running_loss1 += ProtoLoss1.item()          
            seg_proto_running_loss2 += ProtoLoss2.item()          
			
            itr += 1

        i_batch = i_batch + 1
        print('epoch:%d/%d\tCE loss:%g\tEntropy loss:%g\tMP loss:%g\tProto loss1:%g\tProto loss2:%g \n' %  (epoch, cfg.TRAIN_EPOCHS, 
                seg_ce_running_loss/i_batch, seg_entropy_running_loss/i_batch, seg_mp_running_loss/i_batch, seg_proto_running_loss1/i_batch, seg_proto_running_loss2/i_batch))

        #### start testing now
        if epoch % 1 == 0:
            IoUP = test_one_epoch(test_dataset, test_dataloader, net, epoch)
        if IoUP > best_jacc:
            # print('saved!')
            model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,IoUP)),
               old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
            best_jacc = IoUP
            best_epoch = epoch

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
	if now_lr < cfg.TRAIN_LR * 0.01:
		now_lr = cfg.TRAIN_LR * 0.01
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 10*now_lr
	return now_lr

def test_one_epoch(dataset, DATAloader, net, epoch):
	#### start testing now
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
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(DATAloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).cuda()
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				inputs_batched = inputs_batched.cuda()
				_, seg_list, predicts = net(inputs_batched)
				predicts_batched = predicts.clone()
				# seg_batched1 = seg1.clone()
				# seg_batched2 = seg2.clone()
				del predicts
				# del seg1
				# del seg2			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			# seg_batched1 = F.interpolate(seg_batched1, size=None, scale_factor=4, mode='bilinear', align_corners=True)
			# seg_batched2 = F.interpolate(seg_batched2, size=None, scale_factor=4, mode='bilinear', align_corners=True)
			# result_seg1 = torch.argmax(seg_batched1, dim=1).cpu().numpy().astype(np.uint8)
			# result_seg2 = torch.argmax(seg_batched2, dim=1).cpu().numpy().astype(np.uint8)

			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				# p_seg1 = result_seg1[i,:,:]					
				# p_seg2 = result_seg2[i,:,:]					
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
		if epoch % 10 == 0:
			dataset.save_result_train(result_list, cfg.MODEL_NAME)

		return IoUP

if __name__ == '__main__':
	train_net()
