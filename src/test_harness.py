import QNet
import Harness
import io_layer
import numpy as np 
import torch
import mate
import mutate
import correct  
import logging

def test_qnet(config, all_config, in_data):
    print(f"\n*** Candidate {config['name']} ***")
    print(config)
    qnet = QNet.QNet(config, all_config)
    print(f"trainable parameters: {qnet.trainable_parameter_count()}")
    print(f"Input: {in_data[0].shape}")
    result = qnet(in_data[0])
    print(f"Output 0: {result.shape}")
    result = qnet(in_data[1])
    print(f"Output 1: {result.shape}")


logging.basicConfig(level=logging.DEBUG)

io = io_layer.io_layer()

all_conf = io.fetch_all_config()

# Two input test vectors
test_data1 = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))
test_data2 = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))

test_conf0 = io.fetch_config("0")
test_harness = Harness.Harness(test_conf0, all_conf)

in_data = [test_data1, test_data2]

test_harness.start_new_rollout()
action = test_harness.action_for_state(in_data[0])
print(f"Action 0: {action}")
test_harness.set_reward_for_last_action(0.0)

action = test_harness.action_for_state(in_data[1])
print(f"Action 1: {action}")
test_harness.set_reward_for_last_action(5.0)

test_harness.end_rollout()
test_harness.learn_from_replay_buffer()



