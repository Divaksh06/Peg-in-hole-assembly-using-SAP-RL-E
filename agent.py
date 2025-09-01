import torch
import numpy as np
from model import SAP_RL_E
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(self, num_actions=24, learning_rate=1e-3, gamma=0.99, tau=0.005, buffer_size=10000):
        """
        DDQN Agent for Peg-in-Hole task as per PDF
        
        Args:
            num_actions: Number of discrete actions (24 as per PDF)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update parameter for target network
            buffer_size: Size of replay buffer
        """
        self.num_actions = num_actions
        self.lr = learning_rate
        self.g = gamma  # gamma
        self.t = tau    # tau
        self.bs = buffer_size
        
        # Initialize networks
        self.policy_net = SAP_RL_E(num_actions)
        self.target_net = SAP_RL_E(num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Loss weights for multi-task learning (as per PDF)
        self.loss_weights = {'image': 0.1, 'attention': 1.0}
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def selection(self, img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor, bt=1.0):
        """
        Select action using Boltzmann exploration as per PDF
        
        Args:
            img_seq_tensor: Image sequence tensor
            ft_seq_tensor: Force/Torque sequence tensor
            dz_seq_tensor: Displacement sequence tensor
            action_seq_tensor: Previous action sequence tensor
            bt: Boltzmann temperature parameter
            
        Returns:
            action: Selected action
        """
        with torch.no_grad():
            q_values, _, _, _ = self.policy_net(img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor)
            probs = F.softmax(q_values / bt, dim=1).cpu().numpy().flatten()
            action = np.random.choice(self.num_actions, p=probs)
        return action
    
    def learn(self, batch_size):
        """
        Train the agent on a batch of experiences as per PDF methodology
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            loss: Training loss (or None if insufficient data)
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch from replay buffer
        batch_data = self.replay_buffer.sample(batch_size)
        if batch_data is None:
            return None
        
        (img_seqs, ft_seqs, dz_seqs, action_seqs, actions, rewards, 
         next_img_seqs, next_ft_seqs, next_dz_seqs, dones) = batch_data
        
        # Convert to tensors
        img_seqs = torch.tensor(img_seqs, dtype=torch.float32).to(self.device)
        ft_seqs = torch.tensor(ft_seqs, dtype=torch.float32).to(self.device)
        dz_seqs = torch.tensor(dz_seqs, dtype=torch.float32).to(self.device)
        action_seqs = torch.tensor(action_seqs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_img_seqs = torch.tensor(next_img_seqs, dtype=torch.float32).to(self.device)
        next_ft_seqs = torch.tensor(next_ft_seqs, dtype=torch.float32).to(self.device)
        next_dz_seqs = torch.tensor(next_dz_seqs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Forward pass through policy network
        q_values, pred_images, pred_attention, current_attention = self.policy_net(
            img_seqs, ft_seqs, dz_seqs, action_seqs
        )
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1))
        
        # Double DQN target computation (as per PDF)
        with torch.no_grad():
            # Use policy network to select best actions
            next_q_values_policy, _, _, _ = self.policy_net(
                next_img_seqs, next_ft_seqs, next_dz_seqs, action_seqs
            )
            best_next_actions = next_q_values_policy.argmax(1).unsqueeze(1)
            
            # Use target network to evaluate the selected actions
            next_q_values_target, _, _, next_attention_points = self.target_net(
                next_img_seqs, next_ft_seqs, next_dz_seqs, action_seqs
            )
            q_values_of_bna = next_q_values_target.gather(1, best_next_actions)
            
            # Compute target values
            td_target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.g * q_values_of_bna
        
        # Compute losses (as per PDF Equation 2-5)
        loss_q = F.smooth_l1_loss(q_values_for_actions, td_target)  # JQ
        loss_image = F.mse_loss(pred_images, next_img_seqs)  # Ji
        loss_attention = F.mse_loss(pred_attention, next_attention_points)  # Jp
        
        # Total loss (as per PDF Equation 2)
        tl = loss_q + self.loss_weights['image'] * loss_image + self.loss_weights['attention'] * loss_attention
        
        # Optimization step
        self.optimizer.zero_grad()
        tl.backward()
        self.optimizer.step()
        
        # Soft update target network (as per PDF)
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.t * policy_param.data + (1.0 - self.t) * target_param.data)
        
        return tl.item()
    
    def save_model(self, filepath):
        """Save model parameters"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])