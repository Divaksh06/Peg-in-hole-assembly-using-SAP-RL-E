import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax(nn.Module):
    def __init__(self, beta=100):
        super(SoftArgmax, self).__init__()
        self.beta = beta

    def forward(self, heatmaps):
        # heatmaps: [N, num_attention_points, H, W]
        N, num_attention_points, H, W = heatmaps.size()
        # Flatten each heatmap per attention point
        heatmaps_flat = heatmaps.reshape(N * num_attention_points, H * W)
        softmax = F.softmax(heatmaps_flat * self.beta, dim=1)
        softmax = softmax.view(N * num_attention_points, H, W)
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=heatmaps.device),
            torch.arange(W, dtype=torch.float32, device=heatmaps.device),
            indexing='ij'
        )
        # Normalize coordinates to [0, 1]
        y_coords = y_coords / (H - 1)
        x_coords = x_coords / (W - 1)
        # Calculate expected coordinates for each attention point
        expected_x = torch.sum(softmax * x_coords, dim=(1, 2))
        expected_y = torch.sum(softmax * y_coords, dim=(1, 2))
        # Concatenate and reshape: [N, num_attention_points*2]
        attention_points = torch.cat([expected_x, expected_y], dim=0) # [2*N*num_attention_points]
        attention_points = attention_points.view(N, num_attention_points * 2)
        return attention_points

class SAP_RL_E(nn.Module):
    def __init__(self, num_actions=24, num_attention_points=8, window_size=6):
        super(SAP_RL_E, self).__init__()
        self.num_actions = num_actions
        self.num_attention_points = num_attention_points
        self.window_size = window_size

        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # Attention encoder
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # Attention heatmap generation
        self.attention_conv = nn.Conv2d(16, num_attention_points, kernel_size=1)
        self.soft_argmax = SoftArgmax()

        # RL policy network
        self.rl_policy = nn.Sequential(
            nn.Linear(23 * window_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        # Output heads
        self.action_head = nn.Linear(256, num_actions)
        self.pred_attention_head = nn.Linear(256, num_attention_points * 2 * window_size) # Predict attention for 6 timesteps

        # Image decoder
        self.heatmap_fc = nn.Linear(num_attention_points * 2, 64 * 64)
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(8 + num_attention_points, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, img_seq, ft_seq, dz_seq, action_seq):
        """
        Args:
            img_seq: (batch_size, window_len, 3, H, W)
            ft_seq: (batch_size, window_len, 5)
            dz_seq: (batch_size, window_len, 1)
            action_seq: (batch_size, window_len, 1)
        Returns:
            q_values: (batch_size, num_actions)
            pred_images: (batch_size, window_len, 3, H, W)
            pred_attention_points: (batch_size, window_len, num_attention_points*2)
            attention_points: (batch_size, window_len, num_attention_points*2)
        """
        batch_size, window_len, _, H, W = img_seq.shape

        # Flatten sequences for processing
        img_seq_flat = img_seq.view(batch_size * window_len, 3, H, W)

        # Extract image features
        img_features_flat = self.image_encoder(img_seq_flat)
        # Extract attention features
        attn_features_flat = self.attention_encoder(img_seq_flat)
        attn_heatmaps_flat = self.attention_conv(attn_features_flat)
        # Get attention points using soft argmax
        attention_points_flat = self.soft_argmax(attn_heatmaps_flat)
        # Reshape back to sequences
        img_features = img_features_flat.view(batch_size, window_len, 8, H, W)
        attention_points = attention_points_flat.view(batch_size, window_len, self.num_attention_points * 2)

        # Policy input
        policy_input = torch.cat([attention_points, ft_seq, dz_seq, action_seq], dim=-1)  # [batch_size, window_len, 23]
        policy_input_flat = policy_input.view(batch_size, -1)  # [batch_size, 23*window_len]

        rl_hidden = self.rl_policy(policy_input_flat)
        # Outputs
        q_values = self.action_head(rl_hidden)
        pred_attention_points_flat = self.pred_attention_head(rl_hidden) # [batch_size, num_attention_points*2*window_len]
        pred_attention_points = pred_attention_points_flat.view(batch_size, window_len, self.num_attention_points * 2)

        # Generate predicted images
        pred_images = []
        for t in range(window_len):
            # For each timestep in the sequence
            curr_att = pred_attention_points[:, t, :]  # [batch_size, num_attention_points * 2]
            hm = self.heatmap_fc(curr_att)  # [batch_size, 4096]
            hm = hm.view(batch_size, 1, H, W)
            # Repeat heatmap for each attention point
            hm_exp = hm.repeat(1, self.num_attention_points, 1, 1)
            curr_img_feat = img_features[:, t, :, :, :]  # [batch_size, 8, H, W]
            decoder_input = torch.cat([curr_img_feat, hm_exp], dim=1)  # [batch_size, 8+num_attention_points, H, W]
            pred_img = self.image_decoder(decoder_input)  # [batch_size, 3, H, W]
            pred_images.append(pred_img.unsqueeze(1))  # [batch_size, 1, 3, H, W]
        pred_images = torch.cat(pred_images, dim=1)  # [batch_size, window_len, 3, H, W]

        return q_values, pred_images, pred_attention_points, attention_points
