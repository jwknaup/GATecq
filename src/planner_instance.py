#! /venv/bin/python3
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import rospkg
import iolayer
import QNet
import gazebo_simulation
from geometry_msgs.msg import Twist


def calculate_reward(state_pose, goal):
    weight = 1.0
    return weight / np.linalg.norm(state_pose[:-1, :] - goal)


def calculate_control(state):
    # TODO ensure state/input dims match, run forward pass of qnet and parse Q function output into a control
    gain = 1.0
    return gain * np.array([10 - state[1, 0], 0]).reshape((-1, 1))


def main():
    rospy.init_node('rl_planner', anonymous=True)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('gatecq')
    control_publisher = rospy.Publisher("/jackal_velocity_controller/cmd_vel", Twist, queue_size=1)
    vel_cmd = Twist()
    myargv = rospy.myargv(argv=sys.argv)
    print(myargv)
    config_name = myargv[1]
    config_folder = os.path.join(package_path, myargv[2])
    # read in hyper parameters
    io_layer = iolayer.IOLayer(config_folder)
    config = io_layer.fetch_config(config_name)
    all_config = io_layer.fetch_all_config()
    # create Q-network
    qnet = QNet.QNet(config, all_config)
    num_rollouts = 1
    rollout_length = 100
    rollout_dt = 0.05
    # for num rollouts (same world):
    gazebo = gazebo_simulation.GazeboSimulation()
    for ii in range(num_rollouts):
        # reset simulation
        gazebo.reset()
        # start planner
        control_update_rate = rospy.Rate(int(1 / rollout_dt))
        # run planner for N steps and record taus
        taus = []
        state = np.zeros((1, 1))
        action = np.zeros((2, 1))
        goal = np.zeros((2, 1))
        reward_sum = 0
        for jj in range(rollout_length):
            lidar_msg = gazebo.get_laser_scan()
            # print(lidar_msg)
            angles = np.arange(lidar_msg.angle_min, lidar_msg.angle_max, lidar_msg.angle_increment)
            ranges = np.where(np.asarray(lidar_msg.ranges) > lidar_msg.range_max, lidar_msg.range_max, np.asarray(lidar_msg.ranges))
            state_lidar = np.array(ranges).reshape((-1, 1))
            collision = gazebo.get_hard_collision()
            pose_msg = gazebo.get_model_state()
            r = Rotation.from_quat(
                [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z,
                 pose_msg.pose.orientation.w])
            yaw, pitch, roll = r.as_euler('zyx', degrees=False)
            state_pose = np.vstack((pose_msg.pose.position.x, pose_msg.pose.position.y, yaw))
            print(jj, state_pose)
            new_state = np.vstack((state_pose, state_lidar))
            # print(new_state)
            # break if collision
            # calculate reward
            reward = calculate_reward(state_pose, goal)
            # < s, a, r, s'>
            taus.append((state, action, reward, new_state))
            state = new_state.copy()
            # calculate control
            action = calculate_control(state)
            # print(action)
            vel_cmd.linear.x = action[0, 0]
            vel_cmd.angular.z = action[1, 0]
            control_publisher.publish(vel_cmd)
            control_update_rate.sleep()
        # for n in N steps:
        for tau in taus:
            reward_sum += tau[2]
            # grad desc on Q-network
            qnet.backprop(tau)
        print(ii, reward_sum)
    # report final performance
    config['total_reward_test'] = reward_sum
    io_layer.store_config(config)


if __name__ == '__main__':
    main()
