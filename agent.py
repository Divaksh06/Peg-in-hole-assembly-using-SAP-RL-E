import torch
import numpy as np
from model import SAP_RL_E
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(self, num_actions, learning_rate = 1e-3, gamma=0.99, tau=0.005, buffer_size=10000):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.g = gamma
        self.t = tau
        self.bs = buffer_size

        self.policy_net = SAP_RL_E(num_actions)
        self.target_net = SAP_RL_E(num_actions)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.loss_weights = {'image' : 0.1, 'attention' : 1.0}

    def selection(self, state, bt = 1.0):
        with torch.no_grad():
            q_values, _, _, _ = self.policy_net(img_seq_tensor, proprio_seq_sensor)

        probs = F.softmax(q_values/bt, dim = 1).cpu().numpy().flatten()

        action = np.random.choice(self.num_actions, p = probs)

        return action
    
    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None
        
        img_seqs, proprio_seqs, actions, rewards, next_img_seqs, next_proprio_seqs, dones = self.replay_buffer.sample(batch_size)

        q_values, pred_images, pred_attention = self.policy_net(img_seqs, proprio_seqs)

        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values_policy, _, _, _ = self.policy_net(next_img_seqs, next_proprio_seqs)

            best_next_actions = next_q_values_policy.argmax(1).unsqueeze(1)

        next_q_values_target,_,_,next_attention_points = self.target_net(next_img_seqs, next_proprio_seqs)
        q_values_of_bna = next_q_values_target.gather(1,best_next_actions)

        td_target = rewards.unsqueeze(1) + (1-dones.squeeze(1)) * self.g * q_values_of_bna

        loss_q = F.smooth_l1_loss(q_values_for_actions, td_target)
        loss_image = F.mse_loss(pred_attention,next_img_seqs)
        loss_attention = F.mse_loss(pred_attention,next_attention_points)
        tl = loss_q + self.loss_weights['image'] * loss_image + self.loss_weights['attention'] * loss_attention

        self.optimizer.zero_grad()
        tl.backward()
        self.optimizer.step()

        for target_param, policy_param in self.target_net.parameters(), self.policy_net.parameters():
            target_param.data.copy_(self.t * policy_param.data + (1.0 - self.t) * target_param.data)

        return tl.item()