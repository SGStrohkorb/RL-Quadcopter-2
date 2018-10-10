import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 2

        self.state_size = self.action_repeat * 6
        self.action_low = 300
        self.action_high = 700
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([5., 5., 5.])

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        self.x_dist_squared = (self.sim.pose[0] - self.target_pos[0])**2
        self.y_dist_squared = (self.sim.pose[1] - self.target_pos[1])**2
        self.z_dist_squared = (self.sim.pose[2] - self.target_pos[2])**2
        self.total_dist = np.sqrt(self.x_dist_squared + self.y_dist_squared + self.z_dist_squared)
        reward = 5

        self.std = np.std(rotor_speeds)

        reward -= self.std / 1
        '''
        if self.sim.pose[4] >= 0.3:
            reward -= self.sim.pose[4] * 10
        '''
        if self.sim.pose[5] >= 0.3:
            reward -= self.sim.pose[5] * 10
        
        if self.total_dist <= 2.5:
            reward += 10
        
        if self.sim.pose[2] >= 2.5 and self.sim.pose[2] <= 7.5:
            reward += 20

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        #print(rotor_speeds, range(self.action_repeat))
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        #print(pose_all)
        #print()
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state