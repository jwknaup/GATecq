#! /usr/bin/python3.7
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
        self.config_folder = config_folder
        # read in hyper parameters
        self.io_layer = iolayer.IOLayer(config_folder)
        self.config = self.io_layer.fetch_config(config_name)
        all_config = self.io_layer.fetch_all_config()
        self.all_config = all_config
        self.state_dim = all_config['other_inputs'] + all_config['lidar_inputs']
        self.num_actions = all_config['possible_actions']
        # create Q-network
        self.harness = Harness(self.config, all_config)
        self.harness.qnet.load_state_dict(torch.load(os.path.join(config_folder, config_name) + '.pth'))
        self.num_rollouts = 5
        self.rollout_dt = 0.1
        self.motion_primitives = self.config['motion_primitives']
        if self.motion_primitives:
            self.rollout_length = 20
        else:
            self.rollout_length = 100
        self.close_enough = 0.1 #10 cm
        self.final_reward_scale = self.config['final_reward_scale']
        self.leaving_perk = 0.4 # Reward just for leaving the starting place
        self.start = np.array([[-2.0], [3.0]]) #[-2, 3, 1.57]
        self.goal = self.start + np.array([[0.0], [10.0]])
        self.last_distance_to_go = np.linalg.norm(self.start - self.goal)
        # for num rollouts (same world):
        self.gazebo = gazebo_simulation.GazeboSimulation(init_position=self.start.flatten().tolist()+[1.57])

    def action_to_control(self, action):
        velocities = np.linspace(-2.0, 2.0, 4).reshape((1, -1))
        yaw_rates = np.linspace(-1.0, 1.0, 4).reshape((1, -1))
        controls = np.hstack((np.vstack((velocities, np.zeros((1, 4)))), np.vstack((np.zeros((1, 4)), yaw_rates))))
        # controls = np.vstack((velocities, np.zeros((1, 16))))
        return controls[:, action].reshape((-1, 1))

    def action_to_motion_primitive(self, action):
        num_forwards = 5
        num_turns = 15
        straight = np.tile(np.array([[1.0], [0.0]]), (1, num_forwards))
        turn_left = np.tile(np.array([[0.0], [1.0]]), (1, num_turns))
        turn_right = np.tile(np.array([[0.0], [-1.0]]), (1, num_turns))
        move_left_short = np.hstack((turn_left, straight[:, :3], turn_right, straight))
        move_right_short = np.hstack((turn_right, straight[:, :3], turn_left, straight))
        move_left_med = np.hstack((turn_left, straight[:, :], turn_right, straight))
        move_right_med = np.hstack((turn_right, straight[:, :], turn_left, straight))
        move_left_long = np.hstack((turn_left, straight, straight, turn_right, straight))
        move_right_long = np.hstack((turn_right, straight, straight, turn_left, straight))
        motion_primitives = [straight, move_left_short, move_left_med, move_left_long, move_right_short, move_right_med, move_right_long, -straight[:, :2]]
        # ideas: uncertainty aware
        return motion_primitives[action]

    def calculate_reward(self, state_pose):
        distance_to_go = state_pose[7, 0]
        reward = self.last_distance_to_go - distance_to_go
        self.last_distance_to_go = distance_to_go.copy()
        collision_cost = 0.0
        reward -= collision_cost * self.gazebo.get_hard_collision()
        return reward

    def calculate_control(self, state):
        # TODO ensure state/input dims match, run forward pass of qnet and parse Q function output into a control
        state_tensor = torch.from_numpy(state.T).float()
        action = self.harness.action_for_state(state_tensor)
        print(action)
        if self.motion_primitives:
            controls = self.action_to_motion_primitive(action)
        else:
            controls = self.action_to_control(action)
        # print('control guess: ', control)
        gain = 1.0
        pid_control = gain * np.array([self.goal[0, 0] - state[0, 0], 0]).reshape((-1, 1))
        return controls

    def update_action(self, state, action):
        lidar_msg = self.gazebo.get_laser_scan()
        # print(lidar_msg)
        angles = np.arange(lidar_msg.angle_min, lidar_msg.angle_max, lidar_msg.angle_increment)
        ranges = np.where(np.asarray(lidar_msg.ranges) > lidar_msg.range_max, lidar_msg.range_max,
                          np.asarray(lidar_msg.ranges))
        state_lidar = np.array(ranges).reshape((-1, 1))
        # collision = self.gazebo.get_hard_collision()
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
        for ii in range(action.shape[1]):
            vel_cmd = Twist()
            vel_cmd.linear.x = action[0, ii]
            vel_cmd.angular.z = action[1, ii]
            self.control_publisher.publish(vel_cmd)
            rospy.sleep(self.rollout_dt)
        if self.motion_primitives:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
            self.control_publisher.publish(vel_cmd)
            rospy.sleep(self.rollout_dt)
        return tau

    def run_rollouts(self):
        plt.figure
        plt.xlim(-5.5, 1)
        plt.ylim(-1, 15)
        plt.title('path')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        # ax2.set_title('loss')
        plt.plot([-5.5, 1.0], [10.0, 10.0], 'k')
        for ii in range(self.num_rollouts):
            self.harness = Harness(self.config, self.all_config)
            self.harness.qnet.load_state_dict(torch.load(os.path.join(self.config_folder, self.config['name']) + '.pth'))
            # reset simulation
            self.gazebo.reset()
            # start planner
            control_update_rate = rospy.Rate(int(1 / self.rollout_dt))
            # run planner for N steps and record taus
            self.harness.start_new_rollout(0.0)
            state = np.zeros((self.state_dim, 1))
            state[:2, :] = self.start.copy()
            positions = np.zeros((2, self.rollout_length))
            action = self.calculate_control(state)
            reward_sum = 0
            for jj in range(self.rollout_length):
                tau = self.update_action(state, action)
                state, action, reward, new_state = tau
                positions[:, jj] = state[:2, 0]
                state = tau[3].copy()
                action = tau[1].copy()
                # taus[:, jj] = tau
                control_update_rate.sleep()
                print(f"{jj}:{state[:2,0]} -> {tau[2]:.3f}")
                reward_sum += tau[2]
                if state[7, 0] < self.close_enough or state[1, 0] > 10.5:
                    print(f"****** Close enough to goal ***************")
                    reward = 2.0
                    break

            distance_to_goal = state[7,0]
            distance_from_start = state[4,0]
            final_reward = reward + self.final_reward_scale * (10.0 - distance_to_goal)
            print(f"Final reward: {final_reward:.3f}")
            self.harness.set_reward_for_last_action(final_reward)
            self.harness.end_rollout()
            plt.plot(positions[0, :jj+1], positions[1, :jj+1])
            plt.legend(['goal'] + np.arange(ii+1).tolist(), loc="upper right")
            plt.pause(0.5)
            # # for n in N steps:
            # for tau in taus:
            #     reward_sum += tau[2]
            #     # grad desc on Q-network
            #     qnet.backprop(tau)
            # print(ii, reward_sum)
            # Use harness instead
            print(reward_sum)
            # losses = self.harness.learn_from_replay_buffer()
            # ax2.plot(losses, '.')
            # plt.pause(0.5)
        # report final performance
        # self.config['total_reward_test'] = reward_sum
        # I don't think we should be overwriting configs this way
        # self.io_layer.store_config(self.config)
        save_path = os.path.join(self.io_layer.config_folder, self.config['name'])
        plt.savefig(save_path+'_test5', dpi=600)
        # torch.save(self.harness.qnet.state_dict(), save_path+'.pth')


def main():
    planner_instance = Planner()
    planner_instance.run_rollouts()


if __name__ == '__main__':
    main()
