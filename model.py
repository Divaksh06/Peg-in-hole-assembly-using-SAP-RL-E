import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import random
import time
import dataclasses
import PIH_env

class SoftArgmax(nn.Module):
    def __init__(self, beta = 100):
        super(SoftArgmax, self).__init__()
        self.beta = beta

    def forward(self, heatmaps):
        batch_size,_, height, width = heatmaps.size()
        softmax = F.softmax(heatmaps.view(batch_size,-1) * self.beta, dim = 1)
        softmax = softmax.view(batch_size,1,height,width)

        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
        y_coords = y_coords.float().to(heatmaps.device)/(height-1)
        y_coords = y_coords.float().to(heatmaps.device)/(width-1)

        expected_x = torch.sum(softmax*x_coords, dim=(2,3))
        expected_y = torch.sum(softmax*y_coords, dim=(2,3))
        return torch.cat([expected_x, expected_y], dim = 1)
    
class SAP_RL_E(nn.Module):
    def __init__(self, num_actions = 24, num_attention_points = 8):
        super(SAP_RL_E, self).__init__()
        self.num_actions = num_actions
        self.num_attention_points = num_attention_points

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        self.attention_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride = 1, padding = 1), nn.Relu(),
            nn.Conv2d(16, 16, kernel_size=3, stride = 1, padding = 1), nn.Relu(),
        )

        self.attention_conv = nn.conv2d(16, num_attention_points, kernel_size = 1)
        self.soft_argmax = SoftArgmax()

        self.rl_policy = nn.Sequential(
            nn.Linear(23*6,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLu()
        )

        self.action_head = nn.Linear(256, num_actions)

        self.pred_attention_head = nn.Linear(256,num_attention_points*2*6)

        self.heatmap_fc = nn.Linear(num_attention_points*2, 64*64)
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(8+num_attention_points, 16, kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,32,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,2,kernel_size=3,stride=1,padding=1)
        )

    def forward(self, img_seq, proprio_seq):
        batch_size, window_len, _, H, W = img_seq.shape

        all_img_features = img_features
        all_attention_points = attention_points

        img_seq_flat = img_seq.view(batch_size * window_len, 3, H, W)

        img_features_flat = self.image_encoder(img_seq_flat)
        attn_features_flat = self.attention_encoder(img_seq_flat)
        attn_heatmaps_flat = self.attention_conv(attn_features_flat)
        attention_points_flat = self.soft_argmax(attn_heatmaps_flat)

        img_features = img_features_flat.view(batch_size, window_len, 8, H, W)
        attention_points = attention_points_flat.view(batch_size, window_len,self.num_attention_points*2)

        q_values = self.action_head(rl_hidden)

        policy_input = torch.cat([attention_points, proprio_seq], dim=-1)
        policy_input_flat = policy_input.view(batch_size, -1)
        rl_hidden = self.rl_policy(policy_input_flat)

        q_values = self.action_head(rl_hidden)

        pred_attention_points_flat = self.pred_attention_head(rl_hidden)
        pred_attention_points = pred_attention_points_flat.view(batch_size, window_len, self.num_attention_points*2)

        pred_attention_points_flat_reshaped = pred_attention_points.view(batch_size*window_len, -1)
        heatmaps = self.heatmap_fc(pred_attention_points_flat_reshaped).view(batch_size,window_len, self.num_attention_points*2)

        heatmaps = heatmaps.repeat(1,self.num_attention_points,1,1)

        decoder_input = torch.cat([img_features_flat, heatmaps], dim = 1)
        pred_images_flat = self.image_decoder(decoder_input)
        pred_images = pred_images_flat.view(batch_size,window_len,3,H, W)

        return q_values, pred_images, pred_attention_points, attention_points
    
    