import argparse
import pickle
import torch
import numpy as np
from agent import DDQNAgent
from PIH_env import PIHenv
from collections import deque

def create_sequence_from_obs_history(obs_history, window_size=6):
    """
    Create sequence input from observation history as per PDF requirements
    
    Args:
        obs_history: Deque of recent observations
        window_size: Window size (6 as per PDF)
        
    Returns:
        Tuple of tensors for model input
    """
    if len(obs_history) < window_size:
        # Pad with first observation if we don't have enough history
        while len(obs_history) < window_size:
            obs_history.appendleft(obs_history[0])
    
    # Extract sequences
    img_seq = []
    ft_seq = []
    dz_seq = []
    action_seq = []
    
    for i, (obs, action) in enumerate(obs_history):
        img_seq.append(obs['image'])
        ft_seq.append(obs['FT'])
        dz_seq.append(obs['Dz'])
        action_seq.append([action])  # Make it 2D
    
    # Convert to tensors and add batch dimension
    img_seq_tensor = torch.FloatTensor(img_seq).permute(0, 3, 1, 2).unsqueeze(0)  # (1, 6, 3, 64, 64)
    ft_seq_tensor = torch.FloatTensor(ft_seq).unsqueeze(0)  # (1, 6, 5)
    dz_seq_tensor = torch.FloatTensor(dz_seq).unsqueeze(0)  # (1, 6, 1)
    action_seq_tensor = torch.FloatTensor(action_seq).unsqueeze(0)  # (1, 6, 1)
    
    return img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor

def train_agent(episodes=4000, batch_size=32):
    """Train agent using offline data as per PDF methodology"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent
    agent = DDQNAgent(num_actions=24)
    agent.policy_net.to(device)
    agent.target_net.to(device)
    
    print("Loading hole map data...")
    with open('hole_map.pkl', 'rb') as f:
        hole_map_data = pickle.load(f)
    
    # Populate replay buffer
    for episode in hole_map_data:
        if len(episode) > agent.replay_buffer.window_size:
            agent.replay_buffer.push(episode)
    
    print(f"Replay buffer populated with {len(agent.replay_buffer)} episodes.")
    
    # Training loop
    for episode in range(episodes):
        loss = agent.learn(batch_size)
        
        if loss is not None and episode % 50 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
        
        # Save model checkpoints intermittently
        if episode % 1000 == 0:
            agent.save_model(f'sap_rl_e_model_{episode}.pth')
            print(f"Model saved at episode {episode}")

def evaluate_agent(model_path, num_episodes=100):
    """Evaluate trained agent as per PDF methodology"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment and agent
    env = PIHenv()
    agent = DDQNAgent(num_actions=24)
    agent.load_model(model_path)
    agent.policy_net.to(device)
    agent.policy_net.eval()
    
    successes = 0
    total_steps = 0
    window_size = 6
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_history = deque(maxlen=window_size)
        
        # Initialize history with first observation
        for _ in range(window_size):
            obs_history.append((obs, 0))  # (observation, action)
        
        done = False
        steps = 0
        episode_reward = 0
        
        while not done:
            # Create sequence input
            img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor = create_sequence_from_obs_history(obs_history)
            
            # Move to device
            img_seq_tensor = img_seq_tensor.to(device)
            ft_seq_tensor = ft_seq_tensor.to(device)
            dz_seq_tensor = dz_seq_tensor.to(device)
            action_seq_tensor = action_seq_tensor.to(device)
            
            # Select action using low temperature for exploitation
            action = agent.selection(img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor, bt=0.01)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_reward += reward
            
            # Update observation history
            obs_history.append((next_obs, action))
            obs = next_obs
            
            # Check for success (as per PDF: hole found)
            if info.get('inserted', False):
                successes += 1
                break
        
        total_steps += steps
        success_str = 'Success' if info.get('inserted', False) or episode_reward > 50 else 'Failure'
        print(f"Episode {episode+1}: {success_str} in {steps} steps, Reward: {episode_reward:.2f}")
    
    print(f"\nEvaluation Complete:")
    print(f"Success Rate: {successes / num_episodes * 100:.2f}%")
    print(f"Average Steps: {total_steps / num_episodes:.2f}")
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAP-RL-E Agent Train/Eval")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train', help='train or eval mode')
    parser.add_argument('--model_path', type=str, default='sap_rl_e_model_0.pth', help='model path for eval')
    parser.add_argument('--episodes', type=int, default=4000, help='number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--eval_episodes', type=int, default=100, help='number of evaluation episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(episodes=args.episodes, batch_size=args.batch_size)
    elif args.mode == 'eval':
        evaluate_agent(args.model_path, num_episodes=args.eval_episodes)