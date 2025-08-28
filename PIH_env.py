import gymnasium as gym
import pybullet as p
from gymnasium import spaces
import pybullet_data
import numpy as np
import numericalunits

class PIHenv(gym.Env):
    def __init__(self):
        super(PIHenv, self).__init__()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.physics_client = p.connect(p.DIRECT)

        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 1], useFixedBase=True)
        self.hole_id = p.loadURDF("hole.urdf", basePosition=[0, 0, 0], useFixedBase=True)

        self.action_space = spaces.Discrete(12)

        obs_low = -np.ones(16) * np.inf
        obs_high = np.ones(16) * np.inf
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.joint_indices = []
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            # Only revolute joints (type == 0)
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

        self.ee_link_index = None
        for i in range(self.num_joints):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            if link_name in ("wrist_3_link", "ee_link"):
                self.ee_link_index = i
                break

        self.joint_step_size = 0.05

        self.dlim = 4.5 * numericalunits.mm

        self.step_count = 0
        self.max_steps = 200

        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 1], useFixedBase=True)
        self.hole_id = p.loadURDF("hole.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        self.step_count = 0

        for j in self.joint_indices:
            p.resetJointState(self.robot_id, j, targetValue=0.0)

        observation = self._get_observation()
        return observation

    def step(self, action):
        self.step_count += 1

        joint_index = action // 2
        direction = 1 if (action % 2) == 0 else -1

        current_pos = p.getJointState(self.robot_id, self.joint_indices[joint_index])[0]
        new_pos = current_pos + direction * self.joint_step_size

        p.setJointMotorControl2(self.robot_id, self.joint_indices[joint_index],
                                p.POSITION_CONTROL, targetPosition=new_pos)

        p.stepSimulation()

        observation = self._get_observation()
        collision = len(p.getContactPoints(bodyA=self.robot_id)) > 0
        reward = self._compute_reward(collision)
        done = self._check_done(self.step_count, self.max_steps, collision)
        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        joint_positions = [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
        joint_positions = np.array(joint_positions, dtype=np.float32)

        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[4])
        ee_orn = np.array(ee_state[5])

        hole_pos, _ = p.getBasePositionAndOrientation(self.hole_id)
        hole_pos = np.array(hole_pos)

        relative_pos = hole_pos - ee_pos

        observation = np.concatenate([joint_positions, ee_pos, ee_orn, relative_pos])
        return observation.astype(np.float32)

    def _compute_reward(self, collision):
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[4])
        hole_pos, _ = p.getBasePositionAndOrientation(self.hole_id)
        hole_pos = np.array(hole_pos)
        dist = np.linalg.norm(ee_pos - hole_pos)

        if dist > self.dlim:
            reward = -10
        else:
            reward = -dist

        if dist < 0.1:
            reward += 10

        if collision:
            reward -= 1

        return reward

    def _check_done(self, step_count, max_steps, collision):
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[4])
        hole_pos, _ = p.getBasePositionAndOrientation(self.hole_id)
        hole_pos = np.array(hole_pos)
        dist = np.linalg.norm(ee_pos - hole_pos)

        inserted = dist < 0.01 

        if inserted:
            return True
        if step_count >= max_steps:
            return True
        if collision:
            return True

        return False

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.physics_client)







#my code, the above one is my logic but with fixed mistakes


# import gymnasium as gym
# import pybullet as p
# from gymnasium import spaces
# import pybullet_data
# import numpy as np
# import numericalunits


# class PIHenv(gym.Env):
#     def __init__(self):
#         super(PIHenv, self).__init__()
#         self.action_space = spaces.Discrete(12)
#         self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
#         self.physics_client = p.connect(p.DIRECT)
#         self.robot_id = p.loadURDF("ur5.urdf", basePosition = [0,0,1], useFixedBase = True)
#         self.hole_pos = p.loadURDF("hole.urdf", basePosition=[0,0,0], useFixedBase=True)
#         self.joint_indices = []
#         self.num_joints = p.getNumJoints(self.robot_id)
#         for i in self.num_joints:
#             joint_info = p.getJointInfo(self.robot_id, i)
#             joint_index = joint_info[0]
#             joint_name = joint_info[1].decode('utf-8')
#             self.joint_indices.append(joint_index)
#         self.ee_pos = np.array(p.getLinkState(self.robot_id, p.getBodyInfo(self.robot_id)[0])[4])
#         self.ee_pos = np.array(p.getLinkState(self.robot_id, p.getBodyInfo(self.robot_id)[0])[5])
#         self.dlim = 4.5 * numericalunits.mm

#         p.setAdditonalSearchPath(pybullet_data.getDataPath())
#         self.reset()

#     def reset(self):
#         p.resetSimulation
#         observation = self._get_observation()
#         return observation
    
#     def step(self, action):
#         p.stepSimulation()
#         reward = self._compute_reward()
#         done = self._check_done()
#         observation = self._get_observation()
#         return observation, reward, done, {}
    
#     def _get_observation(self):
#         joint_positions = [p.getJointState(self.robot_id, j)[0] for j in range(self.num_joints)]
#         joint_positions = np.array(joint_positions, dtype = np.float32)

#         end_effector_link_index = p.getBodyInfo(self.robot_id)[0]

#         relative_pos = self.hole_pos - self.ee_pos

#         observation = np.concatenate([joint_positions,self.ee_pos,self.ee_orn, relative_pos])
#         return observation
    
#     def _compute_reward(self,collision = False):

#         dist = np.linalg.norm(self.ee_pos - self.hole_pos)
#         if(dist > self.dlim):
#             reward = -10
#         bonus = 10 if dist<0.1 else 0
#         penalty = -1 if collision else 0
#         return reward + penalty + bonus
    
#     def _check_done(self,step_count, max_steps, collision):
#         dist = np.linalg.norm(self.ee_pos-self.hole_pos)
#         inserted = dist < 0.01

#         if inserted:
#             return True
#         if step_count >= max_steps:
#             return True
#         if collision:
#             return True
        
#         return False
