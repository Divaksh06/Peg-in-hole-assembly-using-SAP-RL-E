import pybullet as p
import time
import pybullet_data
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi / 2]))
robot_id = p.loadURDF("ur5.urdf", [0, 0, 0.7],p.getQuaternionFromEuler([0, -np.pi/2, 0]), useFixedBase=True)
eef_link_index = 6
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()