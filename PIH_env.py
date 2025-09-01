import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import numericalunits
import cv2

class PIHenv(gym.Env):
    def __init__(self, render_mode=None, image_size=(64, 64)):
        super(PIHenv, self).__init__()
        
        # Initialize PyBullet
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        if self.physics_client < 0:
            raise RuntimeError("PyBullet physics server issue")
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 1], useFixedBase=True)
        
        # Create hole collision shape (simplified as box)
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.006, 0.006, 0.01]  # 12mm diameter hole approximation
        )
        
        self.hole_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=[0.5, 0.5, 0]  # Position the hole
        )
        
        # Get joint information
        self.joint_indices = []
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        self.num_actionable_joints = len(self.joint_indices)
        
        # Action space: 24 discrete actions (8 directions × 3 step sizes) as per PDF
        self.action_space = spaces.Discrete(24)
        
        # Image size
        self.image_size = image_size
        
        # Find end-effector link
        self.ee_link_index = None
        for i in range(self.num_joints):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            if link_name in ("wrist_3_link", "ee_link"):
                self.ee_link_index = i
                break
        
        if self.ee_link_index is None:
            self.ee_link_index = self.num_joints - 1  # Use last link as fallback
        
        # Parameters from PDF
        self.step_sizes = [0.001, 0.002, 0.003]  # 1, 2, 3 mm step sizes
        self.directions = [
            [1, 0],   # right
            [-1, 0],  # left  
            [0, 1],   # up
            [0, -1],  # down
            [1, 1],   # up-right
            [-1, 1],  # up-left
            [1, -1],  # down-right
            [-1, -1]  # down-left
        ]
        
        self.dlim = 4.5 * numericalunits.mm
        self.step_count = 0
        self.max_steps = 100
        
        # Camera setup for end-effector mounted camera
        self.camera_distance = 0.1  # Close to end-effector
        self.camera_target_offset = [0, 0, -0.05]  # Look slightly ahead
        
        self.reset()
    
    def _get_camera_image(self):
        """Get RGB image from end-effector mounted camera"""
        # Get end-effector pose
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        # Camera position relative to end-effector
        camera_pos = ee_pos + np.array([0, 0, 0.05])  # Slightly above end-effector
        target_pos = ee_pos + np.array(self.camera_target_offset)
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.01,
            farVal=1.0
        )
        
        width, height = self.image_size
        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to RGB array and normalize
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        return rgb_array.astype(np.float32) / 255.0
    
    def _get_forces_torques(self):
        """Get forces and torques from FT sensor"""
        # Get contact forces (simplified simulation of FT sensor)
        contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.ee_link_index)
        
        fx, fy, fz = 0.0, 0.0, 0.0
        mx, my = 0.0, 0.0
        
        for contact in contact_points:
            contact_force = contact[9]  # Normal force
            contact_pos = contact[6]   # Contact position on body B
            
            # Simplified force calculation
            fz += contact_force
            
        # Add some noise to simulate real sensor
        fx += np.random.normal(0, 0.1)
        fy += np.random.normal(0, 0.1)
        fz += np.random.normal(0, 0.5)
        mx += np.random.normal(0, 0.01)
        my += np.random.normal(0, 0.01)
        
        return np.array([fx, fy, fz, mx, my], dtype=np.float32)
    
    def _get_displacement_z(self):
        """Get Z displacement (depth towards wall)"""
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[0])
        
        # Distance from initial Z position (simplified)
        initial_z = 1.0  # Initial robot Z position
        current_z = ee_pos[2]
        dz = initial_z - current_z
        
        return np.array([max(0, dz)], dtype=np.float32)  # Only positive displacement
    
    def _get_observation(self):
        """Get complete observation as per PDF format"""
        image = self._get_camera_image()  # (64, 64, 3)
        ft_data = self._get_forces_torques()  # (5,) - [Fx, Fy, Fz, Mx, My]
        dz_data = self._get_displacement_z()  # (1,) - [Dz]
        
        return {
            'image': image,
            'FT': ft_data,
            'Dz': dz_data
        }
    
    def step(self, action):
        self.step_count += 1
        
        # Parse action: 24 actions = 8 directions × 3 step sizes
        direction_idx = action // 3
        step_size_idx = action % 3
        
        direction = self.directions[direction_idx]
        step_size = self.step_sizes[step_size_idx]
        
        # Get current end-effector position
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        current_pos = np.array(ee_state[0])
        
        # Calculate target position
        target_pos = current_pos.copy()
        target_pos[0] += direction[0] * step_size  # X direction
        target_pos[1] += direction[1] * step_size  # Y direction
        
        # Use inverse kinematics to get joint positions
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos
        )
        
        # Apply joint positions
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[i]
                )
        
        # Step simulation
        for _ in range(10):  # Multiple steps for stability
            p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check for collisions
        contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.ee_link_index)
        collision = len(contact_points) > 0
        
        # Compute reward and check if task is complete
        reward, inserted = self._compute_reward_and_inserted()
        
        # Determine if episode is done
        terminated = inserted
        truncated = (self.step_count >= self.max_steps) or self._out_of_bounds()
        
        info = {
            "collision": collision,
            "inserted": inserted,
            "step_count": self.step_count
        }
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward_and_inserted(self):
        """Compute reward and check if peg is inserted in hole"""
        # Get end-effector position
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[0])
        
        # Get hole position
        hole_pos, _ = p.getBasePositionAndOrientation(self.hole_id)
        hole_pos = np.array(hole_pos)
        
        # Calculate distance
        dist = np.linalg.norm(ee_pos - hole_pos)
        
        # Check if inserted (within threshold)
        inserted = dist < 0.01  # 10mm threshold
        
        # Reward based on distance (as per PDF)
        if inserted:
            reward = 100  # rfoundhole from PDF
        else:
            reward = -1  # Step penalty
            
        return reward, inserted
    
    def _out_of_bounds(self):
        """Check if peg is out of bounds"""
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[0])
        
        hole_pos, _ = p.getBasePositionAndOrientation(self.hole_id)
        hole_pos = np.array(hole_pos)
        
        dist = np.linalg.norm(ee_pos - hole_pos)
        return dist > self.dlim
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Reload environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 1], useFixedBase=True)
        
        # Recreate hole
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.006, 0.006, 0.01]
        )
        
        self.hole_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=[0.5, 0.5, 0]
        )
        
        # Reset step counter
        self.step_count = 0
        
        # Reinitialize joint indices
        self.joint_indices = []
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        # Find end-effector link again
        self.ee_link_index = None
        for i in range(self.num_joints):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            if link_name in ("wrist_3_link", "ee_link"):
                self.ee_link_index = i
                break
        
        if self.ee_link_index is None:
            self.ee_link_index = self.num_joints - 1
        
        # Reset joint positions to random initial configuration
        for j in self.joint_indices:
            joint_info = p.getJointInfo(self.robot_id, j)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            if lower_limit < upper_limit:
                initial_pos = np.random.uniform(lower_limit, upper_limit)
            else:
                initial_pos = np.random.uniform(-0.5, 0.5)
            
            p.resetJointState(self.robot_id, j, targetValue=initial_pos)
        
        # Step simulation to stabilize
        for _ in range(10):
            p.stepSimulation()
        
        observation = self._get_observation()
        info = {"step_count": self.step_count}
        
        return observation, info
    
    def close(self):
        if self.physics_client >= 0:
            p.disconnect(self.physics_client)
            self.physics_client = -1