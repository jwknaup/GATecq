#! /usr/bin/python3.7
import sys
import os
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
        self.num_rollouts = 10
        self.rollout_length = 100
        self.rollout_dt = 0.05
        self.goal = np.array([[10.0], [0.0]])
        # for num rollouts (same world):
        self.gazebo = gazebo_simulation.GazeboSimulation()

    def action_to_control(self, action):
        velocities = np.linspace(-2.0, 2.0, 16).reshape((1, -1))
        yaw_rates = np.linspace(-0.5, 0.5, 8).reshape((1, -1))
        # controls = np.hstack((np.vstack((velocities, np.zeros((1, 8)))), np.vstack((np.zeros((1, 8)), yaw_rates))))
        controls = np.vstack((velocities, np.zeros((1, 16))))
        return controls[:, action].reshape((-1, 1))

    def calculate_reward(self, state_pose):
        # TODO: smarter reward?
        weight = 1.0
        return weight / np.linalg.norm(state_pose[:2, :] - self.goal)

    def calculate_control(self, state):
        # TODO ensure state/input dims match, run forward pass of qnet and parse Q function output into a control
        state_tensor = torch.from_numpy(state.T).float()
        action = self.harness.action_for_state(state_tensor)
        control = self.action_to_control(action)
        print('control guess: ', control)
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
        state_pose = np.vstack((pose_msg.pose.position.x, pose_msg.pose.position.y, yaw))
        new_state = np.vstack((state_pose, state_lidar))
        # print(new_state)
        # break if collision
        # calculate reward
        reward = self.calculate_reward(new_state)
        self.harness.set_reward_for_last_action(reward)
        # < s, a, r, s'>
        tau = (state, action, reward, new_state)
        state = new_state.copy()
        print(state.shape)
        # calculate control
        action = self.calculate_control(state)
        print('control: ', action)
        # print(action)
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0, 0]
        vel_cmd.angular.z = action[1, 0]
        self.control_publisher.publish(vel_cmd)
        return tau

    def run_rollouts(self):
        for ii in range(self.num_rollouts):
            # reset simulation
            self.gazebo.reset()
            # start planner
            control_update_rate = rospy.Rate(int(1 / self.rollout_dt))
            # run planner for N steps and record taus
            self.harness.start_new_rollout()
            taus = []
            state = np.zeros((self.state_dim, 1))
            action = np.zeros((2, 1))
            reward_sum = 0
            for jj in range(self.rollout_length):
                tau = self.update_action(state, action)
                state = tau[3].copy()
                action = tau[1].copy()
                print(state[:2, 0])
                # taus[:, jj] = tau
                control_update_rate.sleep()
                print(jj, tau[2])
                reward_sum += tau[2]
            self.harness.end_rollout()
            # # for n in N steps:
            # for tau in taus:
            #     reward_sum += tau[2]
            #     # grad desc on Q-network
            #     qnet.backprop(tau)
            # print(ii, reward_sum)
            # Use harness instead
            print(reward_sum)
            self.harness.learn_from_replay_buffer()
        # report final performance
        self.config['total_reward_test'] = reward_sum
        self.io_layer.store_config(self.config)


def main():
    planner_instance = Planner()
    planner_instance.run_rollouts()


if __name__ == '__main__':
    main()
