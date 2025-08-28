import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, window_Size = 6):
        self.buffer = deque(max_len = capacity)
        self.window_size = window_Size

    def push(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size)
        img_seqs, proprio_seqs, actions, rewards, next_img_seqs, next_proprio_seqs, dones =[],[] ,[] ,[] ,[] ,[], []

        for episode in sampled_episodes:
            start_idx = random.randint(0,len(episode) - self.window_size-1)
            img_seq = [episode[i]['image'] for i in range(start_idx,start_idx+self.window_size)]
            proprio_seq = [episode[i]['proprio'] for i in range(start_idx, start_idx + self.window_size)]

            action = episode[start_idx + self.window_size - 1]
            reward = episode[start_idx + self.window_size - 1]
            done = episode[start_idx + self.window_size - 1]

            next_img_seq = [episode[i]['image'] for i in range(start_idx,start_idx+self.window_size)]
            next_proprio_seq = [episode[i]['proprio'] for i in range(start_idx,start_idx+self.window_size)]
            img_seqs.append(np.array(img_seq))
            proprio_seqs.append(np.array(proprio_seq))
            next_img_seqs.append(np.array(next_img_seq))
            next_proprio_seqs.append(np.array(next_proprio_seq))
            actions.append(action)
            rewards.append(reward)
            dones.append(done) 

        return img_seqs, proprio_seqs, actions, rewards, next_img_seqs, next_proprio_seqs, dones
        
    def len(self):
        return self.buffer