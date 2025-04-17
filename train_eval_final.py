order_of_string=['x_local','y_local, heading_local', 'velocity', 'acceleration'] 



# Standard library imports
import os
import sys
import csv
import math
import time
import random
import pickle
import statistics
import argparse
from typing import Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch import optim

# Einops imports
import einops
from einops import rearrange, reduce, repeat

# Visualization imports
import matplotlib.pyplot as plt
from PIL import Image

# Local application imports
from TUTR_modified.transformer_encoder import Encoder
from TUTR_modified.transformer_decoder import Decoder
from TUTR_modified.model3 import TrajectoryModel4
from TUTR_modified.utils2 import get_motion_modes_ours



parser = argparse.ArgumentParser() 

parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--obs_len', type=int, default=5)
parser.add_argument('--pred_len', type=int, default=8)
parser.add_argument('--lr_scaling', action='store_true', default=False)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_clusters', type=int, default=50)
parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str, default='NuScene')
parser.add_argument('--lr', type=float, default=.00005)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
parser.add_argument('--dataset_dimension', type=int, default=5)
parser.add_argument('--num_k', type=int, default=10)
parser.add_argument('--ped_num_k', type=int, default=50)
parser.add_argument('--just_x_y', type=bool, default=True)
parser.add_argument('--minADEloss', type=bool, default=True)
parser.add_argument('--lossFunction', type=str, default='minADE',
                    choices=['minADE', 'minADEdiv', 'GMM_NLL'],
                    help='Loss type: minADE (best trajectory), minADEdiv (adds diversity), or GMM_NLL (probabilistic)')
parser.add_argument('--lambda_diversity', type=float, default=0.5, 
                    help='Weight for diversity loss term (used with minADEdiv)')
parser.add_argument('--gmm_eps', type=float, default=1e-6,
                    help='Epsilon value for numerical stability in GMM loss')

args = parser.parse_args()

#! Ensure num_k and ped_num_k are the same for GMM_NLL
if args.lossFunction == 'GMM_NLL':
    #! Use the larger of the two values for both parameters
    max_k = max(args.num_k, args.ped_num_k)
    args.num_k = max_k
    args.ped_num_k = max_k



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
    """
        Original/default loss. Computes the average displacement error (ADE) 
        of the best (closest) predicted trajectory among K candidates.

        Equation:
            ADE = (1/T) * sum_t ||y_t - ŷ_t||_2
            minADE = min_k ADE_k

        Where:
        - y_t is the ground truth at time t
        - ŷ_t is the prediction at time t
        - T is the trajectory length
        - k ∈ [1, K] indexes predicted modes
    """
    def __init__(self):
        super(MinADE_loss, self).__init__()


    def __call__(self, traj, traj_gt):
        """
            Computes average displacement error for the best trajectory in a set, with respect to ground truth
            :param traj: predictions, shape [batch_size, num_modes, sequence_length*2]
            :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
            :return err:  average minADE over batch members
        """
        num_modes = traj.shape[1] #num_k
        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1) # repeat ground truth to have the shape [batch_size, num_modes, sequence_length, 2]
        traj_ = traj.reshape(traj_gt_rpt.shape) # reshape predicted trajectories to have the shape [batch_size, num_modes, sequence_length, 2]

        err = traj_gt_rpt - traj_ # find the difference between ground truth and predicted trajectories 

        # complete this part: calculate the average displacement error between ground truth and each predicted trajectory for each element of batch,
        # find the minimum average displacement error over proposed trajectories for each element of the batch
        # average it over batch members and return it as loss value   

        #! First compute squared error
        squared_err = torch.pow(err, 2).sum(dim=-1)             # sum over x,y dimensions
        #! Compute displacement error for each timestep
        displacement_err = torch.sqrt(squared_err)              # [batch_size, num_modes, sequence_length]
        #! Avg over sequence length to get ADE for each trajectory
        ade_per_mode = torch.mean(displacement_err, dim=-1)     # [batch_size, num_modes]
        #! Find the min ADE over the trajectories for each batch element
        min_ade, _ = torch.min(ade_per_mode, dim=-1)            # [batch_size]
        #! Avg over batch members to get final loss
        final_loss = torch.mean(min_ade)

        return final_loss

class MinADE_with_Diversity_Loss(nn.Module):
    """
        Extension of MinADE that adds a diversity term to penalize similar 
        predicted trajectories and promote coverage of multiple future modes.

        Loss:
            L = minADE + λ * DiversityPenalty

        Diversity Penalty:
            Diversity = 1 / (mean_pairwise_distance + ε)

        Where:
            - Pairwise distances computed between all K predicted modes
            - Encourages spread-out trajectories to better represent multi-modal futures
    """
    def __init__(self, lambda_diversity=0.5, eps=1e-6):
        super(MinADE_with_Diversity_Loss, self).__init__()
        self.lambda_div = lambda_diversity
        self.eps = eps

    def __call__(self, traj, traj_gt):
        """
            traj: [B, K, T*2]
            traj_gt: [B, T, 2]
        """
        B, K, traj_dim = traj.shape
        T = traj_gt.shape[1]

        # -- Reshape --
        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, K, 1, 1)  # [B, K, T, 2]
        traj_reshaped = traj.view(B, K, T, 2)

        # -- MinADE (same as before) --
        err = traj_gt_rpt - traj_reshaped
        squared_err = torch.pow(err, 2).sum(dim=-1)
        displacement_err = torch.sqrt(squared_err)
        ade_per_mode = torch.mean(displacement_err, dim=-1)      # [B, K]
        min_ade, _ = torch.min(ade_per_mode, dim=-1)             # [B]
        ade_loss = torch.mean(min_ade)

        # -- Diversity Loss --
        # Compute pairwise distances between predicted trajectories
        traj_flat = traj_reshaped.view(B, K, -1)                 # [B, K, T*2]
        dists = torch.cdist(traj_flat, traj_flat, p=2)           # [B, K, K]

        # Mask out diagonal (self-distances)
        mask = ~torch.eye(K, dtype=bool).unsqueeze(0).to(traj.device)
        mean_dist = torch.sum(dists * mask.float(), dim=(1, 2)) / (mask.float().sum(dim=(1, 2)) + self.eps)  # [B]
        mean_dist = torch.mean(mean_dist)  # scalar

        # Invert the distance to create a penalty if predictions are too close
        diversity_loss = 1.0 / (mean_dist + self.eps)

        # -- Final loss --
        total_loss = ade_loss + self.lambda_div * diversity_loss
        return total_loss

class GMM_NLL_Loss(nn.Module):
    """
        Probabilistic loss that models predicted trajectories as a Gaussian mixture. 
        Encourages accurate predictions and confident uncertainty estimates by 
        minimizing the negative log-likelihood of the ground truth.

        Equation:
            NLL = -log(Σ_k π_k * N(y | μ_k, Σ_k))

        Where:
            - π_k is the predicted probability (weight) of mode k
            - μ_k is the predicted mean trajectory of mode k
            - Σ_k is the predicted covariance (typically diagonal)
            - N(·) is the Gaussian probability density function
            - y is the ground truth trajectory

        Promotes both accuracy and calibrated uncertainty.
    """
    def __init__(self, eps=1e-6):
        super(GMM_NLL_Loss, self).__init__()
        self.eps = eps

    def __call__(self, traj, scores, gt):
        """
            Adapts GMM NLL to work with the existing model outputs
            
            :param traj: [B, K, T*2] predicted trajectories
            :param scores: [B, K] logits for each mode
            :param gt: [B, T, 2] ground truth trajectory
        """
        B, K, _ = traj.shape
        T = gt.shape[1]
        
        # Reshape trajectories
        traj_reshaped = traj.view(B, K, T, 2)  # [B, K, T, 2]
        gt_expanded = gt.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, T, 2]
        
        # Convert scores to probabilities
        log_pis = F.log_softmax(scores.squeeze(), dim=-1)  # [B, K]
        
        # Compute squared errors for each mode
        squared_err = torch.sum((gt_expanded - traj_reshaped)**2, dim=-1)  # [B, K, T]
        
        # Using fixed variance for simplicity since we don't have predicted sigmas
        fixed_sigma = 0.1  
        
        # Compute gaussian log-likelihood for each timestep and mode
        # -0.5 * (log(2π) + 2*log(σ) + err/σ²)
        log_likelihood = -0.5 * (math.log(2 * math.pi) + 2 * math.log(fixed_sigma) + squared_err / (fixed_sigma**2 + self.eps))
        
        # Sum over timesteps
        log_likelihood = torch.sum(log_likelihood, dim=-1)  # [B, K]
        
        # Combine with log priors
        joint_likelihood = log_pis + log_likelihood  # [B, K]
        
        # LogSumExp for numerical stability when computing log(sum(exp()))
        mixture_likelihood = torch.logsumexp(joint_likelihood, dim=-1)  # [B]
        
        # Negative log-likelihood as loss
        nll_loss = -torch.mean(mixture_likelihood)
        
        return nll_loss



def get_cls_label(gt, motion_modes, soft_label=True):

    # motion_modes [K pred_len 2]
    # gt [B pred_len 2]

    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)  # [B 1 pred_len*2]

    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance = torch.norm(gt - motion_modes, dim=-1)  # [B K]
   
    soft_label = F.softmax(-distance, dim=-1) # [B K]
    
    closest_mode_indices = torch.argmin(distance, dim=-1) # [B]
    
 
    return soft_label, closest_mode_indices

def train(epoch, model, loss_fn, cls_criterion, optimizer, train_dataloader, motion_modes, just_x_y, num_k, ped_num_k, loss_type):
    model.train()
    total_loss = []
    total_batches = len(train_dataloader)
    
    #! Progress tracking
    start_time = time.time()

    for i, (ped_traj, nei_traj, mask_nearest) in enumerate(train_dataloader):
        ped_obs = ped_traj[:, :args.obs_len]

        if just_x_y == True:
            gt = ped_traj[:, args.obs_len:, :2]
        else: 
            gt = ped_traj[:, args.obs_len:]
      
        neis_obs = nei_traj[:, :, :args.obs_len]

        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt.cuda(), motion_modes)

        optimizer.zero_grad()
        
        # Get model predictions
        pred_traj, scores = model(ped_obs.cuda(), neis_obs.cuda(), motion_modes.cuda(), 
                                 mask_nearest.cuda(), closest_mode_indices.cuda(), 
                                 num_k=num_k, ped_num_k=ped_num_k, minADE_loss=True)
        
        # Apply the appropriate loss function
        if loss_type == 'minADE':
            reg_loss = loss_fn(pred_traj, gt.cuda())
        elif loss_type == 'minADEdiv':
            reg_loss = loss_fn(pred_traj, gt.cuda())
        elif loss_type == 'GMM_NLL':
            reg_loss = loss_fn(pred_traj, scores, gt.cuda())
        else:
            reg_loss = loss_fn(pred_traj, gt.cuda())

        # Classification loss for mode probabilities
        clf_loss = cls_criterion(scores.squeeze(), soft_label)

        # Final loss is regression + classification
        loss = reg_loss + clf_loss 

        if torch.isnan(loss):
            sys.exit('Loss became NaN - training stopped')
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.item())

        #! Print progress bar
        progress = (i + 1) / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '▓' * filled_length + '▒' * (bar_length - filled_length)
        
        #! Calculate time stats
        elapsed_time = time.time() - start_time
        if progress > 0:
            est_total_time = elapsed_time / progress
            eta = est_total_time - elapsed_time
        else:
            eta = 0
            
        #! Clear previous line and print updated progress
        print(f"\rEpoch: {epoch} {bar} {progress*100:.1f}%\t| Loss: {loss.item():.4f}\t| ETA: {eta:.1f}s     ", end='')
        
    return total_loss

def test(model, test_dataloader, motion_modes, just_x_y, num_k, ped_num_k):
    model.eval()

    ade = 0
    fde = 0
    num_traj = 0
    ade_vector = torch.tensor([]).cuda()
    fde_vector = torch.tensor([]).cuda()
    
    for i, (ped_traj, nei_traj, mask_nearest) in enumerate(test_dataloader):
        ped_obs = ped_traj[:, :args.obs_len].cuda()
        if just_x_y == True:
            gt = ped_traj[:, args.obs_len:, :2].cuda()
        else: 
            gt = ped_traj[:, args.obs_len:].cuda()

        neis_obs = nei_traj[:, :, :args.obs_len].cuda()

        with torch.no_grad():
            num_traj += ped_obs.shape[0]
            
            pred_trajs, scores = model(ped_obs.cuda(), neis_obs.cuda(), motion_modes.cuda(), 
                                     mask_nearest.cuda(), None, test=True, 
                                     num_k=num_k, ped_num_k=ped_num_k, minADE_loss=True)
            
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

            ade_vector = torch.cat((ade_vector, min_ade), dim=0)
            fde_vector = torch.cat((fde_vector, min_fde), dim=0)
           
            min_ade = torch.sum(min_ade)
            min_fde = torch.sum(min_fde)
            ade += min_ade.item()
            fde += min_fde.item()

    ade = ade / num_traj
    fde = fde / num_traj
    return ade, fde, num_traj, ade_vector, fde_vector



##################################################### Beginning of Main #####################################################



if __name__ == "__main__":

    #! Ensure results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")
    
    #! Check if CUDA is available and print status
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Running on CPU only.")
        print("Warning: Training will be significantly slower without GPU acceleration.")

    #! Save all parser arguments to CSV
    args_dict = vars(args)
    args_file = f"results/{args.lossFunction}_arguments.csv"
    with open(args_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['argument', 'value'])
        for arg, value in args_dict.items():
            writer.writerow([arg, value])
    
    print(f"Arguments saved to {args_file}")

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



    ############################################################################################################################



    motion_modes_file = args.dataset_path + args.dataset_name + str(args.n_clusters) + str(args.dataset_dimension) + str(args.just_x_y) + '_motion_modes.pkl'

    if not os.path.exists(motion_modes_file):
        print('motion modes generating ... ')
        motion_modes = get_motion_modes_ours(dataset_train, args.obs_len, args.pred_len, args.n_clusters, 
                                             args.dataset_path, args.dataset_name, args.dataset_dimension, args.just_x_y)
                                    
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

    if os.path.exists(motion_modes_file):
        print('motion modes loading ... ')
        import pickle
        f = open(args.dataset_path + args.dataset_name +str(args.n_clusters) + str(args.dataset_dimension)
                  + str(args.just_x_y) + '_motion_modes.pkl', 'rb+')
        motion_modes = pickle.load(f)
        f.close()
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()



    #############################################################################################################################



    model = TrajectoryModel4(in_size=args.dataset_dimension, just_x_y= args.just_x_y, obs_len=5, pred_len=8, embed_size=256, 
                             enc_num_layers=2, int_num_layers_list=[2,2], heads=8, forward_expansion=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.cuda()
    model.to(device)



    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Select loss function based on the argument
    if args.lossFunction == 'minADE':
        reg_criterion = MinADE_loss().cuda()
    elif args.lossFunction == 'minADEdiv':
        reg_criterion = MinADE_with_Diversity_Loss(lambda_diversity=args.lambda_diversity).cuda()
    elif args.lossFunction == 'GMM_NLL':
        reg_criterion = GMM_NLL_Loss(eps=args.gmm_eps).cuda()
    else:
        # Default to minADE
        reg_criterion = MinADE_loss().cuda()

    cls_criterion = torch.nn.CrossEntropyLoss().cuda()



    if args.lr_scaling:

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 49], gamma=0.95)



    min_ade = 99
    min_fde = 99

    #! Create training log file
    train_log_file = f"results/{args.lossFunction}_training_log.csv"
    with open(train_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'dataset', 'loss_type', 'total_loss', 'ade', 'fde', 'min_ade', 'min_fde', 'num_traj', 'min_fde_epoch'])

    for ep in range(args.epoch):
        total_loss = train(ep, model, reg_criterion, cls_criterion, optimizer, train_loader, 
                       motion_modes, args.just_x_y, args.num_k, args.ped_num_k, args.lossFunction)
        
        ade, fde, num_traj, ade_vector, fde_vector = test(model, test_loader, motion_modes, 
                                                      args.just_x_y, args.num_k, args.ped_num_k)
        if args.lr_scaling:
            scheduler.step()

        checkpoint_dir = f"{args.checkpoint}_{args.lossFunction}"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        fde_ade_file = f"{checkpoint_dir}/fde_ade"
        
        if min_fde + min_ade > ade + fde:
            min_fde = fde
            min_ade = ade
            min_fde_epoch = ep
            ade_vector_ = np.array(ade_vector.cpu(), copy=True)
            fde_vector_ = np.array(fde_vector.cpu(), copy=True)

            torch.save(model.state_dict(), f"{checkpoint_dir}/best.pth")
            np.savez(fde_ade_file, fde_vector=fde_vector_, ade_vector=ade_vector_)

        train_loss = sum(total_loss) / len(total_loss)

        print(f'\n\tEpoch:{ep}\tdata_set:{args.dataset_name}\tloss_type:{args.lossFunction}\ttotal_loss:{train_loss:.4f}')
        print(f'\tEpoch:{ep}\tade: {ade:.4f}\tfde:{fde:.4f}\tmin_ade: {min_ade:.4f}\tmin_fde:{min_fde:.4f}\tnum_traj: {num_traj:.4f}\tmin_fde_epoch: {min_fde_epoch:.4f}')
        
        #! Append epoch's metrics to the CSV file
        with open(train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, args.dataset_name, args.lossFunction, train_loss, ade, fde, min_ade, min_fde, num_traj, min_fde_epoch])

    print('**************************************************')
    print(f'Best Results for {args.lossFunction}:')

    ade_mean = np.mean(ade_vector_)
    fde_mean = np.mean(fde_vector_)
    ade_median = np.percentile(ade_vector_, 50)
    fde_median = np.percentile(fde_vector_, 50)
    ade_10_percentiles = np.percentile(ade_vector_, 10)
    fde_10_percentiles = np.percentile(fde_vector_, 10)
    ade_90_percentiles = np.percentile(ade_vector_, 90)
    fde_90_percentiles = np.percentile(fde_vector_, 90)
    print('Best Average minADE: ', ade_mean)
    print('Best Average minFDE: ', fde_mean)
    print('Best median minADE: ', ade_median)
    print('Best median minFDE: ', fde_median)
    print('Best 10th percentile minADE: ', ade_10_percentiles)
    print('Best 10th percentile minFDE: ', fde_10_percentiles)
    print('Best 90th percentile minADE: ', ade_90_percentiles)
    print('Best 90th percentile minFDE: ', fde_90_percentiles)

    #! Save best results to a csv file
    best_results_file = f"results/{args.lossFunction}_best_results.csv"
    with open(best_results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['Best Average minADE', ade_mean])
        writer.writerow(['Best Average minFDE', fde_mean])
        writer.writerow(['Best median minADE', ade_median])
        writer.writerow(['Best median minFDE', fde_median])
        writer.writerow(['Best 10th percentile minADE', ade_10_percentiles])
        writer.writerow(['Best 10th percentile minFDE', fde_10_percentiles])
        writer.writerow(['Best 90th percentile minADE', ade_90_percentiles])
        writer.writerow(['Best 90th percentile minFDE', fde_90_percentiles])




    ################################################ Plot the ECDF of ADE and FDE ################################################

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    axs = fig.subplots(1, 2, sharex=True, sharey=True)

    # Cumulative distributions of ADE.
    # We'll use matplotlib's built-in CDF functionality since ecdf might not be available
    axs[0].hist(ade_vector_, bins=50, density=True, cumulative=True, histtype='step',
                label=f"ADE CDF ({args.lossFunction})")

    # cumulative distribution of FDE.
    axs[1].hist(fde_vector_, bins=50, density=True, cumulative=True, histtype='step',
                label=f"FDE CDF ({args.lossFunction})")

    # Label the figure.
    fig.suptitle(f"Cumulative distributions - {args.lossFunction}")
    for ax in axs:
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("min Position Tracking Error over 10 modes (m)")
        ax.set_ylabel("Probability of occurrence")
        ax.label_outer()

    plt.savefig(f"results/{args.lossFunction}_cdf.png")
    plt.show()