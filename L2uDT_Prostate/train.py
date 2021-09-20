#coding=utf-8
import argparse
import os
import time
import logging
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

cudnn.benchmark = True

import numpy as np

import models
from data import datasets
from data.sampler import CycleSampler
from data.data_utils import init_fn
from utils import Parser,criterions

from predict import AverageMeter
from predict import validate_softmax
import setproctitle  # pip install setproctitle

import os
import sys
import torch.nn.functional as F
import cv2
import math
from torch.autograd import Variable
import time

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--cfg', default='DMFNet_GDL_all', required=True, type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)# model_last.pth
parser.add_argument('-Cluster0', '--Cluster0', default=48, type=float, help='Cluster0')
parser.add_argument('-Cluster1', '--Cluster1', default=96, type=float, help='Cluster1')
parser.add_argument('-Cluster2', '--Cluster2', default=192, type=float, help='Cluster2')
parser.add_argument('-Gen_advScale_eps', '--Gen_advScale_eps', default=0.05, type=float, help='Gen_advScale_eps')
parser.add_argument('-Gen_advScale_min', '--Gen_advScale_min', default=0.1, type=float, help='Gen_advScale_min')
parser.add_argument('-Gen_advScale_lr', '--Gen_advScale_lr', default=10., type=float, help='Gen_advScale_lr')
parser.add_argument('-Gen_advScale_iter', '--Gen_advScale_iter', default=10, type=float, help='Gen_advScale_iter')
parser.add_argument('-Gen_advScale_delta', '--Gen_advScale_delta', default=0.05, type=float, help='Gen_advScale_delta')
parser.add_argument('-CL_weight', '--CL_weight', default=0.01, type=float, help='CL_weight')
parser.add_argument('-CL_weight_decay', '--CL_weight_decay', default=0.1, type=float, help='CL_weight_decay')
parser.add_argument('-CL_temp2', '--CL_temp2', default=0.8, type=float, help='CL_temp2')
parser.add_argument('-RMloss_temp', '--RMloss_temp', default=0.5, type=float, help='RMloss_temp')
parser.add_argument('-CL_Region_temp', '--CL_Region_temp', default=0.5, type=float, help='CL_Region_temp')
parser.add_argument('-L2u_temp', '--L2u_temp', default=1., type=float, help='L2u_temp')
parser.add_argument('-num_classes', '--num_classes', default=3, type=int, help='num_classes')

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='ProstateX').add_args(args)
# args.net_params.device_ids= [int(x) for x in (args.gpu).split(',')]
ckpts = args.makedir()

args.resume = False

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def jac(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = torch.sum(output * target, dim=(1,2,3))
    den = torch.sum(output, dim=(1,2,3)) + torch.sum(target, dim=(1,2,3)) - num + eps
    return (1.0 - num/den).mean()

def Jaccard_loss(target, output, eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    # output = nn.Softmax(dim=1)(output)
    loss1 = jac(output[:,1,...],(target==1).float())
    loss2 = jac(output[:,2,...],(target==2).float())
    return loss1+loss2

def EntropyLoss(inputs):
    log_likelihood = torch.log(inputs)
    loss = torch.mean(torch.mul(log_likelihood, inputs))
    return loss

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)

def make_one_hot(x, n_class=2):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    xx[:,1,:,:,:] = (x == 1)
    xx[:,2,:,:,:] = (x == 2)
    return xx.float().cuda()

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
    b, c, d, h, w = feature.size()                            
    delta_P0 = un_P0 - P0              # b * num_clusters * c
    delta_P1 = un_P1 - P1
    delta_out0 = GroupMap0.permute(0, 2, 1).bmm(delta_P0)  ### b * n * c
    delta_out1 = GroupMap1.permute(0, 2, 1).bmm(delta_P1)  ### b * n * c
    delta_out0 = delta_out0.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
    delta_out1 = delta_out1.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
    return delta_out0, delta_out1

def Gen_advScale(inputs, feature, delta_out0, delta_out1, net, epoch):
    f = feature.clone().detach()
    delta0 = delta_out0.clone().detach()
    delta1 = delta_out1.clone().detach()

    eps = args.Gen_advScale_eps
    adv_distance = 1.
    delta_lambda = args.Gen_advScale_delta * (epoch // (args.Gen_advScale_delta*200) + 1)
    lambda_P = torch.ones(args.batch_size, 1, 1, 1) 

    _, _, pred_nobias = net(inputs, feature=f, phase='aug')
    pred_nobias = torch.nn.Upsample(scale_factor=4, mode='trilinear')(pred_nobias)
    pred_nobias = torch.argmax(pred_nobias, dim=1).detach().float()
    i = 0
    while adv_distance > eps:
        lambda_P = to_var(lambda_P)
        feature_l2u = f + lambda_P * (delta0 + delta1)/2. 
        _, _, pred_l2u = net(inputs, feature=feature_l2u, phase='aug')
        pred_l2u = torch.nn.Upsample(scale_factor=4, mode='trilinear')(pred_l2u)

        adv_distance = Jaccard_loss(pred_nobias, torch.softmax(pred_l2u*50., dim=1), eps=1e-7)
        net.zero_grad()
        adv_distance.backward()
        lambda_new = torch.clamp(lambda_P - args.Gen_advScale_lr*lambda_P.grad, min=0.0, max=1.0)
        lambda_P = lambda_new.clone().detach()
        del lambda_new
        i += 1
        if i >= args.Gen_advScale_iter:
            break
    lambda_P = torch.clamp(lambda_P, min=args.Gen_advScale_min, max=args.Gen_advScale_min+delta_lambda)
    # if epoch % 10 == 0:
    #     print(adv_distance, torch.mean(lambda_P))
    return lambda_P

def Gen_prototypes(feature, label, Cluster=None, num_clusters=2):
    b, c, d, h, w = feature.size()                            
    x_t = feature.permute(0, 2, 3, 4, 1)              # b * d * h * w * c
    x_t = x_t.reshape(b, d*h*w, c)              # b * n * c

    label_4down = torch.squeeze(torch.nn.Upsample(scale_factor=1/4, mode='nearest')(torch.unsqueeze(label.float(),1)))
    GroupMap = make_one_hot(label_4down, n_class=args.num_classes)
    GroupMap = GroupMap.reshape(b, args.num_classes, d*h*w)              # b * args.num_classes * n
    if num_clusters != args.num_classes:
        Group_back = Cluster.mul(torch.unsqueeze((label_4down == 0), 1))[:,:num_clusters//3,:,:,:]
        Group_pz = Cluster.mul(torch.unsqueeze((label_4down == 1), 1))[:,num_clusters//3:2*num_clusters//3,:,:,:]
        Group_tz = Cluster.mul(torch.unsqueeze((label_4down == 2), 1))[:,2*num_clusters//3:,:,:,:]
        GroupMap = torch.cat([Group_back, Group_pz, Group_tz], 1)
        GroupMap = GroupMap.permute(0, 2, 3, 4, 1)
        GroupMap = GroupMap.reshape(b, d*h*w, num_clusters).permute(0, 2, 1)              # b * num_clusters * n

    ## Grouping maps & Prototypes
    GroupMap_area = torch.sum(GroupMap, dim=(2), keepdim=True)              # b * num_clusters * 1
    Prototypes = GroupMap.bmm(x_t)              # b * num_clusters * c
    Prototypes = Prototypes/(GroupMap_area + 10e-10)              # b * num_clusters * c
    return GroupMap, Prototypes

def CLLoss(M_zero, M_one):
    Proto_Matrix_sorted = torch.cat([M_zero, M_one], 0)
    dist_mat = Cosine_dist(Proto_Matrix_sorted, temp = args.CL_temp2)
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

def Prototypes_CL(label, Cluster, prototype, Sigma=None):
    b, k, n = Cluster.size()  
    b, k, c = prototype.size()                           
    # Obtain Co-prototypes matrix
    prototype_all = prototype.reshape(b * k, c)
    # prototype = prototype.sum(0)

    Proto_Matrix = prototype_all.mm(prototype_all.permute(1, 0))
    # Proto_Matrix = Cosine_dist(prototype_all, temp = args.CL_temp1)
    Proto_Matrix = Proto_Matrix.reshape(b * k, b * k)

    # Assign labels for prototype according to the prototype_class
    label_4down = torch.squeeze(torch.nn.Upsample(scale_factor=1/4, mode='nearest')(torch.unsqueeze(label.float(),1)))
    label_4down = make_one_hot(label_4down, n_class=args.num_classes).permute(0, 2, 3, 4, 1).reshape(b*n, args.num_classes)
    label_4down = torch.unsqueeze(label_4down, 1)           # (b * n) * 1 * args.num_classes
    group = torch.unsqueeze(Cluster.permute(0, 2, 1).reshape(b*n, k), 2)          # (b * n) * num_clusters * 1
    group = group.bmm(label_4down)          # (b * n) * num_clusters * args.num_classes
    group = group.reshape(b, n, k, args.num_classes)          # (b * n) * num_clusters * args.num_classes
    group = torch.sum(group, dim = (1))
    group = group.reshape(b * k, args.num_classes)          # (b * num_clusters) * args.num_classes
    prototype_zero = torch.squeeze(torch.nonzero(group[:, 0]))
    prototype_one = torch.squeeze(torch.nonzero(group[:, 1]))
    prototype_two = torch.squeeze(torch.nonzero(group[:, 2]))
 
    # Obtain prototype-based InfoNCE loss
    M_zero = torch.index_select(Proto_Matrix, 0, prototype_zero)
    M_one = torch.index_select(Proto_Matrix, 0, prototype_one)
    M_two = torch.index_select(Proto_Matrix, 0, prototype_two)

    loss = CLLoss(M_zero, M_one) + CLLoss(M_zero, M_two) + CLLoss(M_one, M_two)

    return loss/3.

def loss_one_batch(feature, GroupMap_list, pred, label, Sigma):
    ############
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    CE_Loss = criterion(pred, label)
    # CE_Loss += Jaccard_loss(label, pred, eps=1e-7)
    # CE_Loss = CE_Loss/2.
    
    ############
    temp = args.RMloss_temp
    GroupMap0_back = nn.Softmax(dim=1)(GroupMap_list[0] * temp)[:,:args.Cluster0//3,:,:,:]
    GroupMap0_pz = nn.Softmax(dim=1)(GroupMap_list[0] * temp)[:,args.Cluster0//3:2*args.Cluster0//3,:,:,:]
    GroupMap0_tz = nn.Softmax(dim=1)(GroupMap_list[0] * temp)[:,2*args.Cluster0//3:,:,:,:]
    GroupMap1_back = nn.Softmax(dim=1)(GroupMap_list[1] * temp)[:,:args.Cluster1//3,:,:,:]
    GroupMap1_pz = nn.Softmax(dim=1)(GroupMap_list[1] * temp)[:,args.Cluster1//3:2*args.Cluster1//3,:,:,:]
    GroupMap1_tz = nn.Softmax(dim=1)(GroupMap_list[1] * temp)[:,2*args.Cluster1//3:,:,:,:]
    Entropy_loss = EntropyLoss(torch.sum(GroupMap0_back, dim=(0,2,3,4))/(GroupMap0_back.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap0_pz, dim=(0,2,3,4))/(GroupMap0_pz.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap0_tz, dim=(0,2,3,4))/(GroupMap0_tz.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap1_back, dim=(0,2,3,4))/(GroupMap1_back.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap1_pz, dim=(0,2,3,4))/(GroupMap1_pz.sum()+10e-6))
    Entropy_loss += EntropyLoss(torch.sum(GroupMap1_tz, dim=(0,2,3,4))/(GroupMap1_tz.sum()+10e-6))

    GroupMap0 = torch.cat([torch.sum(GroupMap0_back, dim=(1), keepdim=True), torch.sum(GroupMap0_pz, dim=(1), keepdim=True), torch.sum(GroupMap0_tz, dim=(1), keepdim=True)], dim=1)
    GroupMap1 = torch.cat([torch.sum(GroupMap1_back, dim=(1), keepdim=True), torch.sum(GroupMap1_pz, dim=(1), keepdim=True), torch.sum(GroupMap1_tz, dim=(1), keepdim=True)], dim=1)
    label_4down = torch.squeeze(torch.nn.Upsample(scale_factor=1/4, mode='nearest')(torch.unsqueeze(label.float(),1)))
    MP_loss = Jaccard_loss(label_4down, GroupMap0, eps=1e-7)
    MP_loss += Jaccard_loss(label_4down, GroupMap1, eps=1e-7)

    ############
    temp = args.CL_Region_temp
    GroupMap0 = nn.Softmax(dim=1)(GroupMap_list[0] * temp).clone().detach()
    GroupMap1 = nn.Softmax(dim=1)(GroupMap_list[1] * temp).clone().detach()        

    GroupMap0, P0 = Gen_prototypes(feature, label, Cluster=GroupMap0, num_clusters=args.Cluster0)
    GroupMap1, P1 = Gen_prototypes(feature, label, Cluster=GroupMap1, num_clusters=args.Cluster1)

    ProtoLoss0 = Prototypes_CL(label, GroupMap0, P0, Sigma=Sigma[:, :args.Cluster0, :])
    ProtoLoss1 = Prototypes_CL(label, GroupMap1, P1, Sigma=Sigma[:, args.Cluster0:, :])
    return Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1

def train_one_batch(inputs, labels, un_inputs, un_labels, net, optimizer, epoch):
    eye = torch.ones(args.batch_size, args.Cluster0+args.Cluster1, 256)
    Sigma = to_var(eye)

    ##### Actual train #####
    feature, GroupMap_list, pred = net(inputs)
    Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1 = loss_one_batch(feature, GroupMap_list, pred, labels, Sigma)
    loss = CE_Loss + MP_loss + Entropy_loss + args.CL_weight * (ProtoLoss0 + ProtoLoss1) * (args.CL_weight_decay ** int(epoch >= 50)) * (args.CL_weight_decay ** int(epoch >= 100))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ##### Data Augmentation or Semi-supervised #####
    feature, GroupMap_list, pred = net(inputs)
    un_feature, un_GroupMap_list, un_pred = net(un_inputs)
    pseudo_label = torch.from_numpy(torch.argmax(un_pred, dim=1).detach().cpu().numpy()).cuda()

    temp = args.L2u_temp
    GroupMap0 = nn.Softmax(dim=1)(GroupMap_list[0] * temp).clone().detach()
    GroupMap1 = nn.Softmax(dim=1)(GroupMap_list[1] * temp).clone().detach()
    un_GroupMap0 = nn.Softmax(dim=1)(un_GroupMap_list[0] * temp).clone().detach()
    un_GroupMap1 = nn.Softmax(dim=1)(un_GroupMap_list[1] * temp).clone().detach()    

    GroupMap0, P0 = Gen_prototypes(feature, labels, Cluster=GroupMap0, num_clusters=args.Cluster0)
    GroupMap1, P1 = Gen_prototypes(feature, labels, Cluster=GroupMap1, num_clusters=args.Cluster1)
    un_GroupMap0, un_P0 = Gen_prototypes(un_feature, pseudo_label, Cluster=un_GroupMap0, num_clusters=args.Cluster0)
    un_GroupMap1, un_P1 = Gen_prototypes(un_feature, pseudo_label, Cluster=un_GroupMap1, num_clusters=args.Cluster1)

    delta_out0, delta_out1 = L2U_F(feature, P0, P1, GroupMap0, GroupMap1, un_P0, un_P1)
    advScale = Gen_advScale(inputs, feature, delta_out0, delta_out1, net, epoch)
    # advScale = 1.0 * epoch / 50.
    if advScale >= 1.0:
        advScale = 1.0
    l2u_feature = feature + advScale * (delta_out0 + delta_out1)/2.    
    _, _, pred = net(inputs, feature=l2u_feature, phase='aug')
    # pred = torch.nn.Upsample(scale_factor=4, mode='trilinear')(pred)

    Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1 = loss_one_batch(l2u_feature, GroupMap_list, pred, labels, Sigma)
    loss = CE_Loss #+ Entropy_loss + MP_loss #+ args.CL_weight * (ProtoLoss0 + ProtoLoss1) * (args.CL_weight_decay ** int(epoch >= 100)) * (args.CL_weight_decay ** int(epoch >= 150))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    del GroupMap0
    del GroupMap1
    del un_GroupMap0
    del un_GroupMap1

    return Entropy_loss, MP_loss, CE_Loss, ProtoLoss0, ProtoLoss1


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    Network = getattr(models, args.net) #
    model = Network(**args.net_params).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    msg = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'

    msg += '\n' + str(args)
    logging.info(msg)

    # Data loading code
    Dataset = getattr(datasets, args.dataset) #

    train_list = os.path.join(args.train_data_dir, args.train_list)
    unlabel_list = os.path.join(args.train_data_dir, args.unlabel_list)
    test_list = os.path.join(args.valid_data_dir, args.valid_list)
    train_set = Dataset(train_list, root=args.train_data_dir, for_train=True,transforms=args.train_transforms)
    unlabel_set = Dataset(unlabel_list, root=args.train_data_dir, for_train=True,transforms=args.train_transforms)
    test_set = Dataset(test_list, root=args.valid_data_dir, for_train=False,transforms=args.test_transforms)

    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn)
    unlabel_loader = DataLoader(
        dataset=unlabel_set,
        batch_size=args.batch_size,
        collate_fn=unlabel_set.collate,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=test_set.collate,
        num_workers=10,
        pin_memory=True)

    start = time.time()

    enum_batches = len(train_set)/ float(args.batch_size) # nums_batch per epoch

    Entropy_losses = AverageMeter()
    MP_losses = AverageMeter()
    CE_Losses = AverageMeter()
    ProtoLosses1 = AverageMeter()
    ProtoLosses2 = AverageMeter()
    torch.set_grad_enabled(True)

    dice_best = 0.0
    iter_best = 0.0
    # for i_batch, (sample_batched, un_batched) in enumerate(zip(dataloader, un_dataloader)):
    for i, (data, undata) in enumerate(zip(train_loader, unlabel_loader), args.start_iter):
        model.train()
        elapsed_bsize = int( i / enum_batches)+1
        epoch = int((i + 1) / enum_batches)
        setproctitle.setproctitle("Epoch:{}/{}".format(elapsed_bsize,args.num_epochs))

        # actual training
        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)

        data = [t.cuda(non_blocking=True) for t in data]
        x, target = data[:2]
        undata = [t.cuda(non_blocking=True) for t in undata]
        un_x, un_target = undata[:2]

        Entropy_loss, MP_loss, CE_Loss, ProtoLoss1, ProtoLoss2 = train_one_batch(x, target, un_x, un_target, model, optimizer, epoch)

        Entropy_losses.update(Entropy_loss.item(), target.numel())
        MP_losses.update(MP_loss.item(), target.numel())
        CE_Losses.update(CE_Loss.item(), target.numel())
        ProtoLosses1.update(ProtoLoss1.item(), target.numel())
        ProtoLosses2.update(ProtoLoss2.item(), target.numel())

        if (i+1) % args.valid_freq == 0:
            if (i+1) % args.save_freq == 0:
                snap = False #True
                save = False #args.savepath
            else:
                snap = False
                save = False
            msg = 'Iter {0:}, Epoch {1:.4f}, Entropy_losses {2:.7f}, MP_losses {2:.7f}, CE_Losses {2:.7f}, ProtoLosses1 {2:.7f}, ProtoLosses2 {2:.7f}'.format(i+1, \
                    (i+1)/enum_batches, Entropy_losses.avg, MP_losses.avg, CE_Losses.avg, ProtoLosses1.avg, ProtoLosses2.avg)
            logging.info(msg)
            Entropy_losses.reset()
            MP_losses.reset()
            CE_Losses.reset()
            ProtoLosses1.reset()
            ProtoLosses2.reset()

            model.eval()
            with torch.no_grad():
                val_res = validate_softmax(
                                test_loader,
                                model,
                                cfg=args.cfg,
                                savepath=save,
                                save_format = 'nii',
                                names=test_set.names,
                                scoring=True,
                                verbose=True,
                                use_TTA=False,
                                snapshot=snap,
                                postprocess=False,
                                cpu_only=False)
                W_dice = np.around(val_res[0]*100, decimals=3)
                if W_dice > dice_best:
                    old_file = os.path.join(ckpts, 'model_best_iter'+str(iter_best)+'_dice'+str(dice_best)+'.pth')
                    if os.path.exists(old_file) is True:
                        os.remove(old_file) 
                    file_name = os.path.join(ckpts, 'model_best_iter'+str(i+1)+'_dice'+str(W_dice)+'.pth')
                    torch.save({
                        'iter': i+1,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        },
                        file_name)
                    dice_best = W_dice
                    iter_best = i+1
                    logging.info(file_name+' has been saved!')

    i = num_iters + args.start_iter

    msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


if __name__ == '__main__':
    main()
