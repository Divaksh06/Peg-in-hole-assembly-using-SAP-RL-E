import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import random
import math

class PIHenv(gym.Env):
    def __init__(self, render_mode=None, image_size=(64,64)):
        super().__init__()
        # Initialize PyBullet
        if render_mode=="human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        if self.physics_client<0:
            raise RuntimeError("Unable to connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)

        # Spaces and parameters
        self.action_space = spaces.Discrete(24)
        self.image_size = image_size
        self.step_sizes = [0.001,0.002,0.003]
        # 8 directions unit vectors in XY plane
        self.directions = [
            np.array([1,0,0]), np.array([-1,0,0]),
            np.array([0,1,0]), np.array([0,-1,0]),
            np.array([1,1,0])/math.sqrt(2), np.array([-1,1,0])/math.sqrt(2),
            np.array([1,-1,0])/math.sqrt(2), np.array([-1,-1,0])/math.sqrt(2)
        ]
        self.max_steps = 300
        self.step_count = 0
        self.table_height = 0.62

        self._load_environment()
        p.resetDebugVisualizerCamera(1.0,110,-45,[0.5,0,0.5])

    def _load_environment(self):
        # Plane and table
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.4,0,0],
            p.getQuaternionFromEuler([0,0,np.pi/2])
        )

        # Load UR5 with fixed base
        urdf_path = os.path.join(os.getcwd(),"ur5.urdf")
        base_pos = [0.4,0,self.table_height]
        base_ori = p.getQuaternionFromEuler([0,-np.pi/2,0])
        self.robot_id = p.loadURDF(urdf_path, base_pos, base_ori, useFixedBase=True)

        # Fix robot base to table
        p.createConstraint(
            parentBodyUniqueId=self.table_id, parentLinkIndex=-1,
            childBodyUniqueId=self.robot_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
            parentFramePosition=base_pos, childFramePosition=base_pos
        )

        # Identify joints and EE link
        self.joint_indices=[]
        self.ee_link_index=-1
        self.cylinder_joint_index=-1
        for i in range(p.getNumJoints(self.robot_id)):
            info=p.getJointInfo(self.robot_id,i)
            name=info[1].decode()
            if info[2]==p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
            if 'ee_link' in name:
                self.ee_link_index=i
            if 'cylinder_link' in name:
                self.cylinder_joint_index=i
        if self.ee_link_index<0:
            self.ee_link_index=self.joint_indices[-1]

        # Reset first 6 joints to home pose
        home=[0,-1.57,1.57,-1.5,-1.57,0]
        for idx,j in enumerate(self.joint_indices[:6]):
            p.resetJointState(self.robot_id,j,home[idx])

        # Hole placeholder
        self.hole_id=None
        self.target_pos=[0.5,0,self.table_height+0.03]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count=0

        # Reset joints to home pose
        home=[0,-1.57,1.57,-1.5,-1.57,0]
        for idx,j in enumerate(self.joint_indices[:6]):
            p.resetJointState(self.robot_id,j,home[idx])
        for _ in range(100): p.stepSimulation()

        # Spawn hole randomly
        if self.hole_id: p.removeBody(self.hole_id)
        box_urdf=os.path.join(os.getcwd(),"box.urdf")
        x=random.uniform(0.5,0.6)
        y=random.uniform(-0.1,0.1)
        z=self.table_height+0.03
        self.target_pos=[x,y,z]
        self.hole_id=p.loadURDF(box_urdf,self.target_pos,
                                p.getQuaternionFromEuler([0,0,np.pi/2]))

        # Orient cylinder joint to face hole
        if self.cylinder_joint_index>=0:
            ee_pos=np.array(p.getLinkState(self.robot_id,self.cylinder_joint_index)[0])
            vec=self.target_pos - ee_pos
            yaw=math.atan2(vec[1],vec[0])
            pitch=0.0
            # Reset cylinder joint orientation (roll) then apply yaw as base orientation
            p.resetJointState(self.robot_id,self.cylinder_joint_index,0.0)
            orn=p.getQuaternionFromEuler([0,pitch,yaw])
            p.resetBasePositionAndOrientation(self.robot_id,ee_pos,orn)

        return self._get_observation(),{"step_count":0}

    def _get_end_effector_position(self):
        return np.array(p.getLinkState(self.robot_id,self.ee_link_index)[0])

    def _get_camera_image(self):
        ee=self._get_end_effector_position()
        cam_eye=ee+np.array([0,0,0.1])
        cam_target=self.target_pos
        view=p.computeViewMatrix(cam_eye,cam_target,[0,0,1])
        proj=p.computeProjectionMatrixFOV(60,1.0,0.01,2.0)
        w,h=self.image_size
        _,_,px,_,_=p.getCameraImage(w,h,view,proj)
        img=np.reshape(px,(h,w,4))[:,:,:3]
        return img.astype(np.float32)/255.0

    def _get_observation(self):
        return {
            'image':self._get_camera_image(),
            'FT':np.zeros(5,dtype=np.float32),
            'Dz':np.array([max(0,(self.table_height+0.2)-self._get_end_effector_position()[2])],dtype=np.float32)
        }

    def step(self,action):
        self.step_count+=1
        dir_idx=action//3
        step_size=self.step_sizes[0]
        direction=self.directions[dir_idx]
        ee=self._get_end_effector_position()
        target=[ee[0]+direction[0]*step_size,
                ee[1]+direction[1]*step_size,
                ee[2]]
        poses=p.calculateInverseKinematics(self.robot_id,self.ee_link_index,target)
        for i,j in enumerate(self.joint_indices[:6]):
            p.setJointMotorControl2(self.robot_id,j,p.POSITION_CONTROL,
                                    targetPosition=poses[i],force=500)
        for _ in range(10): p.stepSimulation()

        obs=self._get_observation()
        reward,done=self._compute_reward_done()
        collided=self._check_collision()
        terminated=done
        truncated=(self.step_count>=self.max_steps) or collided
        info={'insertion_success':done,'collision':collided,'step_count':self.step_count}
        return obs,reward,terminated,truncated,info

    def _compute_reward_done(self):
        ee=self._get_end_effector_position()
        dxy=np.linalg.norm(ee[:2]-self.target_pos[:2])
        dz=abs(ee[2]-self.target_pos[2])
        done=(dxy<0.01 and dz<0.1)
        reward=100.0 if done else -10*dxy - 5*dz
        return reward,done

    def _check_collision(self):
        tc=p.getContactPoints(self.robot_id,self.table_id)
        hc=p.getContactPoints(self.robot_id,self.hole_id) if self.hole_id else []
        return (len(tc)>0) or (len(hc)>0)

    def close(self):
        if hasattr(self,'physics_client') and self.physics_client>=0:
            p.disconnect(self.physics_client)
            self.physics_client=-1
