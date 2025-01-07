import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image

class RoboticEnvironment:
    def __init__(self, render=False):
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up environment
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot (using a simple UR5 as example)
        self.robot_id = p.loadURDF("ur5/ur5.urdf", [0, 0, 0])
        
        # Camera setup
        self.camera_target = [0, 0, 0]
        self.camera_distance = 1.5
        self.camera_yaw = 50
        self.camera_pitch = -35
        self.camera_height = 480
        self.camera_width = 640
        
    def reset(self):
        """Reset the environment to initial state"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("ur5/ur5.urdf", [0, 0, 0])
        
    def step(self, action):
        """
        Execute one simulation step
        Args:
            action: Robot action to execute
        Returns:
            observation: Current observation after action
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        # Convert action to joint positions/velocities
        target_joint_positions = self._action_to_joint_positions(action)
        
        # Apply action to robot
        for i, pos in enumerate(target_joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=pos
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation
        observation = self.get_observation()
        
        # Calculate reward (example)
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = False
        
        return observation, reward, done, {}
    
    def get_observation(self):
        """Get current observation including camera image and robot state"""
        # Get camera image
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            self.camera_target,
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            0,
            2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        (_, _, px, _, _) = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        # Get robot state
        joint_states = []
        for i in range(p.getNumJoints(self.robot_id)):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])  # Joint position
        
        return {
            'image': rgb_array,
            'joint_states': np.array(joint_states)
        }
    
    def _action_to_joint_positions(self, action):
        """Convert normalized action to joint positions"""
        # Example conversion - modify based on your action space
        action = np.clip(action, -1, 1)
        joint_ranges = [(-np.pi, np.pi)] * 6  # Example for 6-DOF robot
        
        joint_positions = []
        for a, (min_pos, max_pos) in zip(action, joint_ranges):
            pos = min_pos + (a + 1) * 0.5 * (max_pos - min_pos)
            joint_positions.append(pos)
            
        return joint_positions
    
    def _calculate_reward(self):
        """Calculate reward based on task completion"""
        # Implement task-specific reward function
        return 0.0
    
    def close(self):
        """Clean up simulation"""
        p.disconnect()
