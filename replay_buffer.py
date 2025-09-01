import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, window_size=6):
        """
        Replay buffer for storing episode data and sampling sequences as per PDF
        
        Args:
            capacity: Maximum number of episodes to store
            window_size: Length of sequences to sample for training (6 as per PDF)
        """
        self.buffer = deque(maxlen=capacity)
        self.window_size = window_size
    
    def push(self, episode):
        """
        Add an episode to the buffer
        
        Args:
            episode: List of transitions, where each transition is a dict with keys:
                    ['image', 'FT', 'Dz', 'action', 'reward', 'next_image', 'next_FT', 'next_Dz', 'done']
        """
        if len(episode) >= self.window_size + 1:  # Need at least window_size + 1 transitions
            self.buffer.append(episode)
    
    def sample(self, batch_size):
        """
        Sample a batch of sequences from the buffer as per PDF requirements
        
        Args:
            batch_size: Number of sequences to sample
            
        Returns:
            Tuple of (img_seqs, ft_seqs, dz_seqs, action_seqs, actions, rewards, 
                     next_img_seqs, next_ft_seqs, next_dz_seqs, dones)
        """
        if len(self.buffer) < batch_size:
            return None
        
        sampled_episodes = random.sample(list(self.buffer), batch_size)
        
        img_seqs = []
        ft_seqs = []
        dz_seqs = []
        action_seqs = []
        actions = []
        rewards = []
        next_img_seqs = []
        next_ft_seqs = []
        next_dz_seqs = []
        dones = []
        
        for episode in sampled_episodes:
            # Ensure we have enough transitions
            max_start_idx = len(episode) - self.window_size
            if max_start_idx <= 0:
                continue
                
            start_idx = random.randint(0, max_start_idx - 1)
            
            # Extract sequences of length window_size (6 as per PDF)
            img_seq = []
            ft_seq = []
            dz_seq = []
            action_seq = []
            next_img_seq = []
            next_ft_seq = []
            next_dz_seq = []
            
            # Build sequences
            for i in range(start_idx, start_idx + self.window_size):
                img_seq.append(episode[i]['image'])
                ft_seq.append(episode[i]['FT'])
                dz_seq.append(episode[i]['Dz'])
                action_seq.append([episode[i]['action']])  # Make it 2D for consistency
                
                next_img_seq.append(episode[i]['next_image'])
                next_ft_seq.append(episode[i]['next_FT'])
                next_dz_seq.append(episode[i]['next_Dz'])
            
            # Get action, reward, done from the last transition in the window
            last_transition = episode[start_idx + self.window_size - 1]
            
            img_seqs.append(np.array(img_seq))
            ft_seqs.append(np.array(ft_seq))
            dz_seqs.append(np.array(dz_seq))
            action_seqs.append(np.array(action_seq))
            next_img_seqs.append(np.array(next_img_seq))
            next_ft_seqs.append(np.array(next_ft_seq))
            next_dz_seqs.append(np.array(next_dz_seq))
            
            actions.append(last_transition['action'])
            rewards.append(last_transition['reward'])
            dones.append(1.0 if last_transition['done'] else 0.0)
        
        return (
            np.array(img_seqs),
            np.array(ft_seqs),
            np.array(dz_seqs),
            np.array(action_seqs),
            np.array(actions),
            np.array(rewards),
            np.array(next_img_seqs),
            np.array(next_ft_seqs),
            np.array(next_dz_seqs),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)