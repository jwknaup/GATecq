import QNet
import Harness
import io_layer
import numpy as np 
import torch
import logging

# See log messages
logging.basicConfig(level=logging.DEBUG)

io = io_layer.io_layer()
all_conf = io.fetch_all_config()

# Two input test vectors
test_data1 = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))
test_data2 = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))
in_data = [test_data1, test_data2]

# Fetch a configuration
test_conf0 = io.fetch_config("0")

# Create a harness for Deep Q learning
harness = Harness.Harness(test_conf0, all_conf)

# Do a rollout
harness.start_new_rollout()
action = harness.action_for_state(in_data[0])
print(f"Action 0: {action}")
harness.set_reward_for_last_action(-1.0)
action = harness.action_for_state(in_data[1])
print(f"Action 1: {action}")
harness.set_reward_for_last_action(5.0)
harness.end_rollout()

# Learn from the replay buffer
harness.learn_from_replay_buffer()

