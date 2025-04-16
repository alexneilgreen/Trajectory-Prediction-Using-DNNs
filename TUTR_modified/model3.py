import torch
import torch.nn as nn
import sys
from TUTR_modified.transformer_encoder import Encoder
from TUTR_modified.transformer_decoder import Decoder



class TrajectoryModel4(nn.Module):

    def __init__(self, in_size, just_x_y, obs_len, pred_len, embed_size, enc_num_layers, int_num_layers_list, heads, forward_expansion):
        super(TrajectoryModel4, self).__init__()
        self.just_x_y = just_x_y
        if self.just_x_y == True:
            out_size = 2

            self.embedding= nn.Linear(in_size*(obs_len)+out_size*(pred_len), embed_size)


            self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=False)
            self.cls_head = nn.Linear(embed_size, 1)
            self.cls_head2 = nn.Linear(embed_size, 1)

            self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
            self.social_decoder =  Decoder(embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False)
            self.reg_head = nn.Linear(embed_size, out_size*pred_len)

    def spatial_interaction(self, ped, neis, mask):
        
        # ped [B K embed_size]
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]
        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]
        int_feat = self.social_decoder(ped, nei_embeddings, mask)  # [B K embed_size]
        scores_nei = self.cls_head2(int_feat)
        soft=nn.Softmax(dim=-1)
        scores_nei = soft(scores_nei.squeeze())
        return int_feat, scores_nei # [B K embed_size]
    
    def forward(self, ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=False,num_k=20, ped_num_k =100, minADE_loss = True):


        if self.just_x_y == True:
            ped_obs = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1) 
            # [B K obs_len 2]
            motion_modes = motion_modes.unsqueeze(0).repeat(ped_obs.shape[0], 1, 1, 1)
            ped_obs = ped_obs.reshape(ped_obs.shape[0], ped_obs.shape[1], -1)  # [B K obs_len*5]
            motion_modes = motion_modes.reshape(motion_modes.shape[0], motion_modes.shape[1], -1)


            input_embedder=torch.cat((ped_obs , motion_modes ), dim=-1)
            ped_embedding=self.embedding(input_embedder)

            ped_feat = self.mode_encoder(ped_embedding)  # [B K embed_size]
            scores = self.cls_head(ped_feat).squeeze()  # [B K]

            if not test and not minADE_loss:
                top_k_indices = torch.topk(scores, k=ped_num_k, dim=-1).indices  # [B num_k]
                top_k_indices = top_k_indices.flatten()  # [B*num_k]
                index1 = torch.LongTensor(range(ped_feat.shape[0])).cuda()  # [B]
                index1 = index1.unsqueeze(1).repeat(1, ped_num_k).flatten() # [B*num_k]
                index2 = top_k_indices # [B*num_k]
                top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
                top_k_feat = top_k_feat.reshape(ped_feat.shape[0], ped_num_k, -1)  # [B num_k embed_size]
                int_feats, scores_nei = self.spatial_interaction(top_k_feat, neis_obs, mask)
                index3 = torch.LongTensor(range(scores_nei.shape[0])).cuda()
                index4 = torch.argmax(scores_nei.squeeze(), dim=-1)
                closest_feat = int_feats[index3, index4].squeeze(1)
                pred_trajs = self.reg_head(closest_feat)
                return pred_trajs,scores_nei

            if not test and minADE_loss:

                top_k_indices = torch.topk(scores, k=ped_num_k, dim=-1).indices  # [B num_k]
                top_k_indices = top_k_indices.flatten()  # [B*num_k]
                index1 = torch.LongTensor(range(ped_feat.shape[0])).cuda()  # [B]
                index1 = index1.unsqueeze(1).repeat(1, ped_num_k).flatten() # [B*num_k]
                index2 = top_k_indices # [B*num_k]
                top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
                top_k_feat = top_k_feat.reshape(ped_feat.shape[0], ped_num_k, -1)  # [B num_k embed_size]
                int_feats, scores_nei = self.spatial_interaction(top_k_feat, neis_obs, mask)

                top_k_indices2 = torch.topk(scores_nei.squeeze(), k=num_k, dim=-1).indices  # [B num_k]
                top_k_indices2 = top_k_indices2.flatten()  # [B*num_k]
                index5 = torch.LongTensor(range(int_feats.shape[0])).cuda()  # [B]
                index5 = index5.unsqueeze(1).repeat(1, num_k).flatten() # [B*num_k]
                index6 = top_k_indices2 # [B*num_k]
                top_k_feat2 = int_feats[index5, index6]  # [B*num_k embed_size]
                top_k_feat2 = top_k_feat2.reshape(int_feats.shape[0], num_k, -1)  # [B num_k embed_size]
                pred_trajs = self.reg_head(top_k_feat2)

                return pred_trajs,scores_nei

            if test:
                top_k_indices = torch.topk(scores, k=ped_num_k, dim=-1).indices  # [B num_k]
                top_k_indices = top_k_indices.flatten()  # [B*num_k]
                index1 = torch.LongTensor(range(ped_feat.shape[0])).cuda()  # [B]
                index1 = index1.unsqueeze(1).repeat(1, ped_num_k).flatten() # [B*num_k]
                index2 = top_k_indices # [B*num_k]
                top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
                top_k_feat = top_k_feat.reshape(ped_feat.shape[0], ped_num_k, -1)  # [B num_k embed_size]
                int_feats, scores_nei = self.spatial_interaction(top_k_feat, neis_obs, mask)

                top_k_indices2 = torch.topk(scores_nei.squeeze(), k=num_k, dim=-1).indices  # [B num_k]
                top_k_indices2 = top_k_indices2.flatten()  # [B*num_k]
                index5 = torch.LongTensor(range(int_feats.shape[0])).cuda()  # [B]
                index5 = index5.unsqueeze(1).repeat(1, num_k).flatten() # [B*num_k]
                index6 = top_k_indices2 # [B*num_k]
                top_k_feat2 = int_feats[index5, index6]  # [B*num_k embed_size]
                top_k_feat2 = top_k_feat2.reshape(int_feats.shape[0], num_k, -1)  # [B num_k embed_size]
                pred_trajs = self.reg_head(top_k_feat2)

                return pred_trajs,scores_nei

