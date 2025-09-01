import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax(nn.Module):
    def __init__(self, beta=100):
        super(SoftArgmax, self).__init__()
        self.beta = beta
    
    def forward(self, heatmaps):
        batch_size, _, height, width = heatmaps.size()
        
        # Apply softmax to get attention distribution
        softmax = F.softmax(heatmaps.contiguous().view(batch_size, -1) * self.beta, dim=1)
        softmax = softmax.view(batch_size, 1, height, width)
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=heatmaps.device),
            torch.arange(width, dtype=torch.float32, device=heatmaps.device),
            indexing='ij'
        )
        
        # Normalize coordinates to [0, 1]
        y_coords = y_coords / (height - 1)
        x_coords = x_coords / (width - 1)
        
        # Compute expected coordinates
        expected_x = torch.sum(softmax * x_coords, dim=(2, 3))
        expected_y = torch.sum(softmax * y_coords, dim=(2, 3))
        
        return torch.cat([expected_x, expected_y], dim=1)

class SAP_RL_E(nn.Module):
    def __init__(self, num_actions=24, num_attention_points=8):
        super(SAP_RL_E, self).__init__()
        
        self.num_actions = num_actions
        self.num_attention_points = num_attention_points
        
        # Image encoder for feature extraction (as per PDF Fig. 1)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        
        # Attention encoder (as per PDF Fig. 1)
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        
        # Attention heatmap generation
        self.attention_conv = nn.Conv2d(16, num_attention_points, kernel_size=1)
        self.soft_argmax = SoftArgmax()
        
        # RL policy network (as per PDF)
        # Input: attention points (8*2*6) + FT (5*6) + Dz (1*6) + actions (1*6) = (16+5+1+1)*6 = 23*6 = 138
        self.rl_policy = nn.Sequential(
            nn.Linear(23*6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        
        # Output heads
        self.action_head = nn.Linear(256, num_actions)
        self.pred_attention_head = nn.Linear(256, num_attention_points * 2 * 6)  # Predict attention for 6 timesteps
        
        # Image decoder (as per PDF Fig. 1)
        self.heatmap_fc = nn.Linear(num_attention_points * 2, 64 * 64)
        
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(8 + num_attention_points, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)  # Output 3 channels for RGB
        )
    
    def forward(self, img_seq, ft_seq, dz_seq, action_seq):
        """
        Forward pass of the SAP-RL-E model as per PDF
        
        Args:
            img_seq: (batch_size, window_len, 3, H, W) - Image sequences
            ft_seq: (batch_size, window_len, 5) - Force/Torque sequences
            dz_seq: (batch_size, window_len, 1) - Displacement sequences  
            action_seq: (batch_size, window_len, 1) - Previous action sequences
            
        Returns:
            q_values: (batch_size, num_actions) - Q-values for actions
            pred_images: (batch_size, window_len, 3, H, W) - Predicted next images
            pred_attention_points: (batch_size, window_len, num_attention_points*2) - Predicted attention points
            attention_points: (batch_size, window_len, num_attention_points*2) - Current attention points
        """
        batch_size, window_len, _, H, W = img_seq.shape
        
        # Flatten sequences for processing
        img_seq_flat = img_seq.view(batch_size * window_len, 3, H, W)
        
        # Extract image features (as per PDF)
        img_features_flat = self.image_encoder(img_seq_flat)
        
        # Extract attention features (as per PDF)
        attn_features_flat = self.attention_encoder(img_seq_flat)
        attn_heatmaps_flat = self.attention_conv(attn_features_flat)
        
        # Get attention points using soft argmax (as per PDF)
        attention_points_flat = self.soft_argmax(attn_heatmaps_flat)
        
        # Reshape back to sequences
        img_features = img_features_flat.view(batch_size, window_len, 8, H, W)
        attention_points = attention_points_flat.view(batch_size, window_len, self.num_attention_points * 2)
        
        # Combine attention points with proprioceptive data (as per PDF)
        # attention_points: (batch_size, window_len, 16)
        # ft_seq: (batch_size, window_len, 5) 
        # dz_seq: (batch_size, window_len, 1)
        # action_seq: (batch_size, window_len, 1)
        policy_input = torch.cat([attention_points, ft_seq, dz_seq, action_seq], dim=-1)
        policy_input_flat = policy_input.view(batch_size, -1)  # (batch_size, 23*6)
        
        # RL policy forward pass
        rl_hidden = self.rl_policy(policy_input_flat)
        
        # Generate outputs
        q_values = self.action_head(rl_hidden)
        pred_attention_points_flat = self.pred_attention_head(rl_hidden)
        pred_attention_points = pred_attention_points_flat.view(batch_size, window_len, self.num_attention_points * 2)
        
        # Generate predicted images (as per PDF)
        # Use predicted attention points to create heatmaps
        pred_attention_flat = pred_attention_points.view(batch_size * window_len, -1)
        heatmaps = self.heatmap_fc(pred_attention_flat).view(batch_size * window_len, 1, H, W)
        
        # Repeat heatmaps for each attention point
        heatmaps_expanded = heatmaps.repeat(1, self.num_attention_points, 1, 1)
        
        # Combine image features with attention heatmaps
        decoder_input = torch.cat([img_features_flat, heatmaps_expanded], dim=1)
        
        # Decode to predicted images
        pred_images_flat = self.image_decoder(decoder_input)
        pred_images = pred_images_flat.view(batch_size, window_len, 3, H, W)
        
        return q_values, pred_images, pred_attention_points, attention_points