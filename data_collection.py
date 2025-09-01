import pickle
from PIH_env import PIHenv
import numpy as np

def collect_data(num_episodes=50, max_steps_per_ep=100):
    """
    Collect training data from the environment using random policy as per PDF methodology
    """
    env = PIHenv()
    hole_map_data = []
    
    for episode in range(num_episodes):
        print(f"Collecting data for episode {episode+1}/{num_episodes}")
        
        obs, info = env.reset()  # obs is now a dict with 'image', 'FT', 'Dz'
        current_episode_data = []
        prev_action = 0  # Initialize previous action
        
        for step in range(max_steps_per_ep):
            # Sample random action (0-23 as per PDF)
            action = env.action_space.sample()
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Create transition data as per PDF format
            transition = {
                'image': obs['image'],  # Current image observation (64x64x3)
                'FT': obs['FT'],  # Current force/torque data (5,)
                'Dz': obs['Dz'],  # Current displacement data (1,)
                'action': prev_action,  # Previous action for this observation
                'reward': reward,
                'next_image': next_obs['image'],  # Next image observation
                'next_FT': next_obs['FT'],  # Next force/torque data
                'next_Dz': next_obs['Dz'],  # Next displacement data
                'done': terminated or truncated
            }
            
            current_episode_data.append(transition)
            obs = next_obs
            prev_action = action  # Update previous action for next step
            
            if terminated or truncated:
                break
        
        hole_map_data.append(current_episode_data)
    
    env.close()
    
    # Save collected data
    with open('hole_map.pkl', 'wb') as f:
        pickle.dump(hole_map_data, f)
    
    print(f"Hole map data saved to hole_map.pkl")
    print(f"Collected {len(hole_map_data)} episodes with total transitions: {sum(len(ep) for ep in hole_map_data)}")

if __name__ == '__main__':
    collect_data()