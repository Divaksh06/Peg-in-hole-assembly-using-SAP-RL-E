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