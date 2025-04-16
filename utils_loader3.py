order_of_string=['x_local','y_local, heading_local', 'x_global', 'y_golbal', 'heading_global', 'velocity', 'acceleration'] #0:3 and 6::
note_string=['different agents are in 3rd dimension, in interpolation vector 0 means less than 3 consecutive nans, 1 means 3 or more interpolated values']

import os
import pickle
import random
import time
import numpy as np
import torch
import statistics
import math
import sys
import einops
import argparse
import torch.nn.functional as F
from TUTR_modified.transformer_encoder import Encoder
from TUTR_modified.transformer_decoder import Decoder

from TUTR_modified.model3 import TrajectoryModel4
from TUTR_modified.utils2 import get_motion_modes_ours
from einops import rearrange, reduce, repeat

from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt

from PIL import Image
from torch import optim
import torch.nn as nn
from typing import Tuple


import pickle



parser = argparse.ArgumentParser() 

parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--obs_len', type=int, default=5)
parser.add_argument('--pred_len', type=int, default=8)
parser.add_argument('--lr_scaling', action='store_true', default=False)
#parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_clusters', type=int, default=50)
parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str, default='NuScene')
parser.add_argument('--lr', type=float, default=.00005)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
parser.add_argument('--dataset_dimension', type=int, default=5)
parser.add_argument('--num_k', type=int, default=10)
parser.add_argument('--ped_num_k', type=int, default=50)
parser.add_argument('--just_x_y', type=bool, default=True)
parser.add_argument('--minADEloss', type=bool, default=True)




args = parser.parse_args()



class Custom_Dataset(torch.utils.data.dataset.Dataset):#, histsize=None, futsize=None):
    def __init__(self, _dataset,traj_path):
        #Custom_Dataset(train_set,trajpath,bottomview_image_path)
        self.dataset = _dataset
        self.traj_path = traj_path


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample_ann = self.dataset[index]
        file_name = self.traj_path +'/'+ str(sample_ann)

        sample = np.load(file_name +'.npz')
        target_traj= sample['target_traj']
        nei_traj= sample['nei_traj']
        mask_nearest= sample['mask_nearest']
        category= np.array(sample['category'])


        return torch.FloatTensor(target_traj),\
                torch.FloatTensor(nei_traj),\
                torch.FloatTensor(mask_nearest),\
            
                

class MinADE_loss(nn.Module):
  def __init__(self):
    super(MinADE_loss, self).__init__();

  
  def __call__(self, traj, traj_gt):
    """
    Computes average displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length*2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    traj_ = traj.reshape(traj_gt_rpt.shape)

    err = traj_gt_rpt - traj_
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err, dim=2) /traj_.shape[2]
    err, inds = torch.min(err, dim=1)
    err= torch.mean(err)

    return err



# beginning of main

# Open the pickle file in binary mode for reading
file_name = './dataset/Nuscenes_data' +'/'+ 'Train_Val_Sets'
sets = np.load(file_name +'.npz')
train_set = sets['train_set']
test_set = sets['val_set']
print('len(train_set)',len(train_set))
print('len(test_set)',len(test_set))
trajpath_train = './dataset/Nuscenes_data/train'
trajpath_test = './dataset/Nuscenes_data/test'
batch_size=32
# just_x_y=False
dataset_train=Custom_Dataset(train_set,trajpath_train)
train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=4)#,collate_fn=collate_fn2)
dataset_test=Custom_Dataset(test_set,trajpath_test)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size,shuffle=True,num_workers=4)



#######################################################################################################


motion_modes_file = args.dataset_path + args.dataset_name +str(args.n_clusters)+ str(args.dataset_dimension) + str(args.just_x_y) + '_motion_modes.pkl'

if not os.path.exists(motion_modes_file):
    print('motion modes generating ... ')
    motion_modes = get_motion_modes_ours(dataset_train, args.obs_len, args.pred_len, args.n_clusters, args.dataset_path, args.dataset_name, args.dataset_dimension, args.just_x_y)
                                   
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()


if os.path.exists(motion_modes_file):
    print('motion modes loading ... ')
    import pickle
    f = open(args.dataset_path + args.dataset_name +str(args.n_clusters)+ str(args.dataset_dimension) + str(args.just_x_y) + '_motion_modes.pkl', 'rb+')
    motion_modes = pickle.load(f)
    f.close()
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()



##############################################################################################################




model = TrajectoryModel4(in_size=args.dataset_dimension, just_x_y= args.just_x_y, obs_len=5, pred_len=8, embed_size=256, 
enc_num_layers=2, int_num_layers_list=[2,2], heads=8, forward_expansion=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.cuda()
model.to(device)




optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
if args.minADEloss:
    reg_criterion = MinADE_loss().cuda()
else:
    reg_criterion = torch.nn.SmoothL1Loss().cuda()

cls_criterion = torch.nn.CrossEntropyLoss().cuda()




if args.lr_scaling:

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 99], gamma=0.95)

   

def get_cls_label(gt, motion_modes, soft_label=True):

    # motion_modes [K pred_len 2]
    # gt [B pred_len 2]

    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)  # [B 1 pred_len*2]

    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance = torch.norm(gt - motion_modes, dim=-1)  # [B K]
   
    soft_label = F.softmax(-distance, dim=-1) # [B K]
    
    closest_mode_indices = torch.argmin(distance, dim=-1) # [B]
    
 
    return soft_label, closest_mode_indices



def train(epoch, model, reg_criterion, cls_criterion, optimizer, train_dataloader, motion_modes,just_x_y,num_k,ped_num_k,minADE_loss):
    model.train()
    total_loss = []

    for i, (ped_traj,nei_traj,mask_nearest) in enumerate(train_dataloader):


        ped_obs = ped_traj[:, :args.obs_len]

        if just_x_y == True:
            gt = ped_traj[:, args.obs_len:,:2]
        else: 
            gt = ped_traj[:, args.obs_len:]
      
        neis_obs = nei_traj[:, :, :args.obs_len]

        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt.cuda(), motion_modes)


        optimizer.zero_grad()
        pred_traj, scores = model(ped_obs.cuda(), neis_obs.cuda(), motion_modes.cuda(),mask_nearest.cuda(), closest_mode_indices.cuda(), num_k=num_k, ped_num_k =ped_num_k,minADE_loss=minADE_loss)
        # print('pred_traj.shape',pred_traj.shape)
        # print('gt.shape',gt.shape)
        if args.minADEloss:

            reg_loss = reg_criterion(pred_traj, gt.cuda())
        else:
            reg_label = gt.reshape(pred_traj.shape).cuda()
            reg_loss = reg_criterion(pred_traj, reg_label)

        clf_loss = cls_criterion(scores.squeeze(), soft_label)

        loss = reg_loss + clf_loss 

        if torch.isnan(loss):
            sys.exit('hello65')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.item())
        # sys.exit('line 581')
    return total_loss







def test(model, test_dataloader, motion_modes, just_x_y,num_k,ped_num_k):

    model.eval()

    ade = 0
    fde = 0
    num_traj = 0
    ade_vector =torch.tensor([]).cuda()
    fde_vector =torch.tensor([]).cuda()
    # ade_vector.cuda()
    # fde_vector.cuda()
    
    for i, (ped_traj,nei_traj,mask_nearest) in enumerate(test_dataloader):
        #ped_traj=rearrange(ped_traj, 'b c h -> b h c')
        #nei_traj=rearrange(nei_traj, 'b c h w -> b w h c')

        ped_obs = ped_traj[:, :args.obs_len].cuda()
        if just_x_y == True:

            gt = ped_traj[:, args.obs_len:,:2].cuda()
        else: 
            gt = ped_traj[:, args.obs_len:].cuda()

        neis_obs = nei_traj[:, :, :args.obs_len].cuda()

        with torch.no_grad():

            num_traj += ped_obs.shape[0]
            pred_trajs, scores = model(ped_obs.cuda(), neis_obs.cuda(), motion_modes.cuda(), mask_nearest.cuda(), None, test=True, num_k= num_k, ped_num_k=ped_num_k,minADE_loss=True)
            # top_k_scores = torch.topk(scores, k=20, dim=-1).values
            # top_k_scores = F.softmax(top_k_scores, dim=-1)
            if just_x_y:

                pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

            else:
                pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], args.dataset_dimension)

            gt_ = gt.unsqueeze(1)
            norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1)
            fde_ = norm_[:, :, -1]
            min_ade, min_ade_index = torch.min(ade_, dim=-1)
            min_fde, min_fde_index = torch.min(fde_, dim=-1)

            # vis_predicted_trajectories(ped_obs, gt, pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[-2], -1), top_k_scores,
            #                             min_fde_index)

            # b-ade/fde
            # batch_index = torch.LongTensor(range(top_k_scores.shape[0])).cuda() 
            # min_ade_p = top_k_scores[batch_index, min_ade_index]
            # min_fde_p = top_k_scores[batch_index, min_fde_index]
            # min_ade = min_ade + (1 - min_ade_p)**2
            # min_fde = min_fde + (1 - min_fde_p)**2
            # print('min_ade.shape',min_ade.shape)
            ade_vector=torch.cat((ade_vector,min_ade),dim=0)
            fde_vector=torch.cat((fde_vector,min_fde),dim=0)
           
            min_ade = torch.sum(min_ade)
            min_fde = torch.sum(min_fde)
            ade += min_ade.item()
            fde += min_fde.item()

    ade = ade / num_traj
    fde = fde / num_traj
    return ade, fde, num_traj, ade_vector, fde_vector







min_ade = 99
min_fde = 99

for ep in range(args.epoch):

    total_loss = train(ep, model, reg_criterion, cls_criterion, optimizer, train_loader, motion_modes,args.just_x_y, args.num_k, args.ped_num_k,args.minADEloss)
    
    ade, fde, num_traj, ade_vector, fde_vector = test(model, test_loader, motion_modes, args.just_x_y, args.num_k, args.ped_num_k)
    if args.lr_scaling:
        scheduler.step()

    if not os.path.exists(args.checkpoint +'_'+ args.dataset_name +'_n_clusters_' + str(args.n_clusters)  + '_num_K_' +str(args.num_k) + '_minADE_Loss_' + str(args.minADEloss)):
        os.makedirs(args.checkpoint +'_'+ args.dataset_name +'_n_clusters_' + str(args.n_clusters)  + '_num_K_' +str(args.num_k) + '_minADE_Loss_' + str(args.minADEloss))
    fde_ade_file = args.checkpoint +'_'+ args.dataset_name +'_n_clusters_' + str(args.n_clusters)  + '_num_K_' +str(args.num_k) + '_minADE_Loss_' + str(args.minADEloss) +'/' + 'fde_ade'
    if min_fde + min_ade > ade + fde:
        min_fde = fde
        min_ade = ade
        min_fde_epoch = ep
        ade_vector_ = np.array(ade_vector.cpu())
        fde_vector_ = np.array(fde_vector.cpu())

        torch.save(model.state_dict(), args.checkpoint +'_'+ args.dataset_name +'_n_clusters_' + str(args.n_clusters)  + '_num_K_' +str(args.num_k) + '_minADE_Loss_' + str(args.minADEloss) + '/best.pth')  # OK
        np.savez(fde_ade_file, fde_vector= fde_vector_, ade_vector=ade_vector_)

    train_loss = sum(total_loss) / len(total_loss)

    print('epoch:', ep, 'data_set:', args.dataset_name, 'total_loss:', train_loss)
    print('epoch:', ep, 'ade:', ade, 'fde:', fde, 'min_ade:', min_ade, 'min_fde:', min_fde, 'num_traj:', num_traj,
          "min_fde_epoch:", min_fde_epoch)

print('**************************************************')
print('Best Results:')

ade_mean = np.mean(ade_vector_)
fde_mean =np.mean(fde_vector_)
ade_median = np.percentile(ade_vector_, 50)
fde_median = np.percentile(fde_vector_, 50)
ade_10_percentiles = np.percentile(ade_vector_, 10)
fde_10_percentiles = np.percentile(fde_vector_, 10)
ade_90_percentiles = np.percentile(ade_vector_, 90)
fde_90_percentiles = np.percentile(fde_vector_, 90)
print('Best Average minADE: ',ade_mean)
print('Best Average minFDE: ',fde_mean)
print('Best median minADE: ',ade_median)
print('Best median minFDE: ',fde_median)
print('Best 10th percentile minADE: ',ade_10_percentiles)
print('Best 10th percentile minFDE: ',fde_10_percentiles)
print('Best 90th percentile minADE: ',ade_90_percentiles)
print('Best 90th percentile minFDE: ',fde_90_percentiles)




fig = plt.figure(figsize=(9, 4), layout="constrained")
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# Cumulative distributions of ADE.
axs[0].ecdf(ade_vector_, label="ADE CDF")



# cumulative distribution of FDE.
axs[1].ecdf(fde_vector_, label="FDE CDF")


# Label the figure.
fig.suptitle("Cumulative distributions")
for ax in axs:
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("min Position Tracking Error over 10 modes (m)")
    ax.set_ylabel("Probability of occurrence")
    ax.label_outer()

plt.show()





