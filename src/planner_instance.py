#! /usr/bin/python3.8
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import rospy
import rospkg
import iolayer
import QNet
import gazebo_simulation
from geometry_msgs.msg import Twist
from Harness import Harness

class Planner:
    def __init__(self):
        rospy.init_node('rl_planner', anonymous=True)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('gatecq')
        self.control_publisher = rospy.Publisher("/jackal_velocity_controller/cmd_vel", Twist, queue_size=1)
        args = rospy.myargv(argv=sys.argv)
        print(args)
        config_name = args[1]
        config_folder = os.path.join(package_path, args[2])
        # read in hyper parameters
        self.io_layer = iolayer.IOLayer(config_folder)
        self.config = self.io_layer.fetch_config(config_name)
        all_config = self.io_layer.fetch_all_config()
        self.state_dim = all_config['other_inputs'] + all_config['lidar_inputs']
        self.num_actions = all_config['possible_actions']
        # create Q-network
        self.harness = Harness(self.config, all_config)
        self.num_rollouts = 100
        self.rollout_length = 250
        self.rollout_dt = 0.05
        self.close_enough = 0.1 #10 cm
        self.final_reward_scale = 3.0
        self.leaving_perk = 0.4 # Reward just for leaving the starting place
        self.goal = np.array([[10.0], [0.0]])
        self.start = np.array([[0.0], [0.0]])
        self.last_distance_to_go = np.linalg.norm(self.start - self.goal)
        # for num rollouts (same world):
        self.gazebo = gazebo_simulation.GazeboSimulation()

    def action_to_control(self, action):
        velocities = np.linspace(-2.0, 2.0, 8).reshape((1, -1))
        yaw_rates = np.linspace(-0.5, 0.5, 8).reshape((1, -1))
        controls = np.hstack((np.vstack((velocities, np.zeros((1, 8)))), np.vstack((np.zeros((1, 8)), yaw_rates))))
        # controls = np.vstack((velocities, np.zeros((1, 16))))
        return controls[:, action].reshape((-1, 1))

    def calculate_reward(self, state_pose):
        distance_to_go =  state_pose[7, 0]
        reward = self.last_distance_to_go - distance_to_go
        self.last_distance_to_go = distance_to_go
        return reward

    def calculate_control(self, state):
        # TODO ensure state/input dims match, run forward pass of qnet and parse Q function output into a control
        state_tensor = torch.from_numpy(state.T).float()
        action = self.harness.action_for_state(state_tensor)
        control = self.action_to_control(action)
        # print('control guess: ', control)
        gain = 1.0
        pid_control = gain * np.array([self.goal[0, 0] - state[0, 0], 0]).reshape((-1, 1))
        return control

    def update_action(self, state, action):
        lidar_msg = self.gazebo.get_laser_scan()
        # print(lidar_msg)
        angles = np.arange(lidar_msg.angle_min, lidar_msg.angle_max, lidar_msg.angle_increment)
        ranges = np.where(np.asarray(lidar_msg.ranges) > lidar_msg.range_max, lidar_msg.range_max,
                          np.asarray(lidar_msg.ranges))
        state_lidar = np.array(ranges).reshape((-1, 1))
        collision = self.gazebo.get_hard_collision()
        pose_msg = self.gazebo.get_model_state()
        r = Rotation.from_quat(
            [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z,
             pose_msg.pose.orientation.w])
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        
        position = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y])
        delta_to_start = self.start[:,0] - position
        distance_to_start = np.linalg.norm(delta_to_start)
        angle_to_start = math.atan2(delta_to_start[1], delta_to_start[0])
        relative_angle_to_start = angle_to_start - yaw
       
        delta_to_goal = self.goal[:,0] - position
        distance_to_goal = np.linalg.norm(delta_to_goal)
        angle_to_goal = math.atan2(delta_to_goal[1], delta_to_goal[0])
        relative_angle_to_goal = angle_to_goal - yaw

        # print(f"yaw:{yaw/math.pi:.2f} pi radians, angle to start:{relative_angle_to_start/math.pi:.2f} pi radians, angle to goal:{relative_angle_to_goal/math.pi:.2f} pi radians")
        
        state_pose = np.vstack((pose_msg.pose.position.x, pose_msg.pose.position.y, math.cos(relative_angle_to_start), math.sin(relative_angle_to_start), distance_to_start , math.cos(relative_angle_to_goal), math.sin(relative_angle_to_goal), distance_to_goal))        
        # state_pose = np.vstack((pose_msg.pose.position.x, pose_msg.pose.position.y, math.cos(yaw), math.sin(yaw)))
        new_state = np.vstack((state_pose, state_lidar))
        # break if collision
        # calculate reward
        reward = self.calculate_reward(new_state)
        self.harness.set_reward_for_last_action(reward)
        # < s, a, r, s'>
        tau = (state, action, reward, new_state)
        state = new_state.copy()
        # print(state.shape)
        # calculate control
        action = self.calculate_control(state)
        # print('control: ', action)
        # print(action)
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0, 0]
        vel_cmd.angular.z = action[1, 0]
        self.control_publisher.publish(vel_cmd)
        return tau

    def run_rollouts(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlim(-10, 12)
        ax1.set_ylim(-10, 10)
        ax1.set_title('path')
        ax2.set_title('loss')
        ax1.plot(10.0, 0.0, 'ok')
        for ii in range(self.num_rollouts):
            # reset simulation
            self.gazebo.reset()
            # start planner
            control_update_rate = rospy.Rate(int(1 / self.rollout_dt))
            # run planner for N steps and record taus
            self.harness.start_new_rollout()
            taus = []
            state = np.zeros((self.state_dim, 1))
            positions = np.zeros((2, self.rollout_length))
            action = np.zeros((2, 1))
            reward_sum = 0
            for jj in range(self.rollout_length):
                tau = self.update_action(state, action)
                state, action, reward, new_state = tau
                positions[:, jj] = state[:2, 0]
                state = tau[3].copy()
                action = tau[1].copy()
                # print(state[:2, 0])
                # taus[:, jj] = tau
                control_update_rate.sleep()
                print(f"{jj}:{state[:2,0]} -> {tau[2]:.3f}")
                reward_sum += tau[2]
                if state[7, 0] < self.close_enough:
                    print(f"****** Close enough to goal. Exiting ***************")
                    break

            distance_to_goal = state[7,0]
            distance_from_start = state[4,0]
            final_reward = self.leaving_perk * distance_from_start + self.final_reward_scale * (10.0 - distance_to_goal)
            print(f"Final reward: {final_reward:.3f}")
            self.harness.replace_reward_for_last_action(final_reward)
            self.harness.end_rollout()
            ax1.plot(positions[0, :], positions[1, :])
            fig.legend(['goal'] + np.arange(ii+1).tolist(), loc="upper right")
            plt.pause(0.1)
            # # for n in N steps:
            # for tau in taus:
            #     reward_sum += tau[2]
            #     # grad desc on Q-network
            #     qnet.backprop(tau)
            # print(ii, reward_sum)
            # Use harness instead
            print(reward_sum)
            losses = self.harness.learn_from_replay_buffer()
            ax2.plot(losses)
            plt.pause(0.1)
        # report final performance
        self.config['total_reward_test'] = reward_sum
        # I don't think we should be overwriting configs this way
        # self.io_layer.store_config(self.config)


def main():
    planner_instance = Planner()
    planner_instance.run_rollouts()


if __name__ == '__main__':
    main()
