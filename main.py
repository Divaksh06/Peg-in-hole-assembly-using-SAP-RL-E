import argparse
import pickle
import torch
import numpy as np
import os
from agent import DDQNAgent
from PIH_env import PIHenv
from collections import deque

def create_sequence_from_obs_history(obs_history, window_size=6):
    """Create sequence input from observation history"""
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
    """Train agent using offline data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent
    agent = DDQNAgent(num_actions=24)
    agent.policy_net.to(device)
    agent.target_net.to(device)

    # Check if training data exists
    if not os.path.exists('hole_map.pkl'):
        print("ERROR: hole_map.pkl not found!")
        print("Please run data collection first: python data_collection_enhanced.py")
        return

    print("Loading hole map data...")
    try:
        with open('hole_map.pkl', 'rb') as f:
            hole_map_data = pickle.load(f)
        print(f"Loaded {len(hole_map_data)} episodes")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Populate replay buffer
    episodes_added = 0
    for episode in hole_map_data:
        if len(episode) >= agent.replay_buffer.window_size:
            agent.replay_buffer.push(episode)
            episodes_added += 1

    print(f"Replay buffer populated with {episodes_added} episodes.")

    if episodes_added == 0:
        print("ERROR: No valid episodes in training data!")
        return

    # Training loop
    for episode in range(episodes):
        loss = agent.learn(batch_size)

        if loss is not None and episode % 50 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
        elif loss is None and episode % 100 == 0:
            print(f"Episode {episode}, No learning (insufficient data)")

        # Save model checkpoints
        if episode % 1000 == 0:
            model_path = f'sap_rl_e_model_{episode}.pth'
            agent.save_model(model_path)
            print(f"Model saved: {model_path}")

    # Save final model
    final_path = 'sap_rl_e_model_final.pth'
    agent.save_model(final_path)
    print(f"Final model saved: {final_path}")

def evaluate_agent(model_path, num_episodes=100, show_gui=True):
    """Evaluate trained agent with enhanced environment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model file '{model_path}' not found!")
        print("Creating random agent for evaluation...")
        use_trained_model = False
    else:
        print(f"Loading model from: {model_path}")
        use_trained_model = True

    # Initialize enhanced environment with GUI option
    try:
        render_mode = "human" if show_gui else None
        env = PIHenv(render_mode=render_mode)
        print(f"Enhanced environment initialized successfully (GUI: {'ON' if show_gui else 'OFF'})")
    except Exception as e:
        print(f"ERROR initializing environment: {e}")
        return

    agent = DDQNAgent(num_actions=24)
    if use_trained_model:
        try:
            agent.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Using random agent instead")
            use_trained_model = False

    agent.policy_net.to(device)
    agent.policy_net.eval()

    successes = 0
    collisions = 0
    total_steps = 0
    total_rewards = 0
    window_size = 6

    print(f"\nStarting evaluation with {'trained' if use_trained_model else 'random'} agent...")
    print("Environment features:")
    print("  ✓ Robot on table setup")
    print("  ✓ Randomized hole placement")
    print("  ✓ Enhanced collision detection")
    print("  ✓ Camera view from robot")

    for episode in range(num_episodes):
        try:
            obs, info = env.reset()
            print(f"\nEpisode {episode+1}: Environment reset successful")

            obs_history = deque(maxlen=window_size)

            # Initialize history with first observation
            for _ in range(window_size):
                obs_history.append((obs, 0))  # (observation, action)

            done = False
            steps = 0
            episode_reward = 0

            while not done and steps < 300:  # Increased step limit
                # Create sequence input
                img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor = create_sequence_from_obs_history(obs_history)

                # Move to device
                img_seq_tensor = img_seq_tensor.to(device)
                ft_seq_tensor = ft_seq_tensor.to(device)
                dz_seq_tensor = dz_seq_tensor.to(device)
                action_seq_tensor = action_seq_tensor.to(device)

                # Select action
                action = agent.selection(img_seq_tensor, ft_seq_tensor, dz_seq_tensor, action_seq_tensor, bt=1.0)

                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                episode_reward += reward

                # Update observation history
                obs_history.append((next_obs, action))
                obs = next_obs

                # Check for success or collision
                if info.get('insertion_success', False):
                    successes += 1
                    print(f"  SUCCESS! Hole found in {steps} steps")
                    break
                elif info.get('collision', False):
                    collisions += 1
                    print(f"  COLLISION detected at step {steps}")
                    break

            total_steps += steps
            total_rewards += episode_reward

            # Episode summary
            if info.get('insertion_success', False):
                result = 'Success'
            elif info.get('collision', False):
                result = 'Collision'
            else:
                result = 'Timeout'

            print(f"Episode {episode+1}: {result} in {steps} steps, Reward: {episode_reward:.2f}")

        except Exception as e:
            print(f"ERROR in episode {episode+1}: {e}")
            continue

    # Final results
    if num_episodes > 0:
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE - Enhanced Environment")
        print(f"{'='*60}")
        print(f"Success Rate: {successes / num_episodes * 100:.2f}%")
        print(f"Collision Rate: {collisions / num_episodes * 100:.2f}%")
        print(f"Average Steps: {total_steps / num_episodes:.2f}")
        print(f"Average Reward: {total_rewards / num_episodes:.2f}")
        print(f"Total Episodes: {num_episodes}")
        print(f"Model Used: {'Trained' if use_trained_model else 'Random'}")

    try:
        env.close()
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAP-RL-E Agent with Enhanced Environment")
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval', help='train or eval mode')
    parser.add_argument('--model_path', type=str, default='sap_rl_e_model_final.pth', help='model path for eval')
    parser.add_argument('--episodes', type=int, default=4000, help='number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--eval_episodes', type=int, default=10, help='number of evaluation episodes')
    parser.add_argument('--gui', action='store_true', help='show PyBullet GUI during evaluation')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(episodes=args.episodes, batch_size=args.batch_size)
    elif args.mode == 'eval':
        evaluate_agent(args.model_path, num_episodes=args.eval_episodes, show_gui=args.gui)
