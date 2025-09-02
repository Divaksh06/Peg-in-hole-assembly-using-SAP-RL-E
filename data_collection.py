import pickle
from PIH_env import PIHenv
import numpy as np

def collect_data(num_episodes=50, max_steps_per_ep=200, show_gui=True):
    """
    Collect training data using enhanced environment with table setup
    """
    # Create environment with GUI enabled
    render_mode = "human" if show_gui else None
    env = PIHenv(render_mode=render_mode)

    hole_map_data = []

    for episode in range(num_episodes):
        print(f"Collecting data for episode {episode+1}/{num_episodes}")

        obs, info = env.reset()  # obs is now a dict with 'image', 'FT', 'Dz'
        current_episode_data = []
        prev_action = 0  # Initialize previous action

        for step in range(max_steps_per_ep):
            # Sample random action (0-23)
            action = env.action_space.sample()

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Create transition data
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
                if info.get('insertion_success', False):
                    print(f"  SUCCESS! Insertion achieved in {step+1} steps")
                elif info.get('collision', False):
                    print(f"  Episode ended due to collision at step {step+1}")
                break

        print(f"Episode {episode+1} collected {len(current_episode_data)} transitions")
        hole_map_data.append(current_episode_data)

    env.close()

    # Save collected data
    with open('hole_map.pkl', 'wb') as f:
        pickle.dump(hole_map_data, f)

    print(f"\nHole map data saved to hole_map.pkl")
    print(f"Collected {len(hole_map_data)} episodes with total transitions: {sum(len(ep) for ep in hole_map_data)}")

    # Print statistics
    successful_episodes = sum(1 for ep in hole_map_data if len(ep) > 0 and any(t.get('done', False) and not t.get('collision', True) for t in ep))
    print(f"Episodes with potential success: {successful_episodes}")
    print(f"Average episode length: {np.mean([len(ep) for ep in hole_map_data]):.1f} transitions")

if __name__ == '__main__':
    # Set show_gui=True to see PyBullet simulation with table setup
    collect_data(show_gui=True)
