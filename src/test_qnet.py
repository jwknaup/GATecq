import QNet
import io_layer
import numpy as np 
import torch

all_conf = io_layer.fetch_all_config()

test_data = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))

test_conf = io_layer.fetch_config("0")
qnet = QNet.QNet(test_conf, all_conf)
print(f"Input 0: {test_data.shape}")
result = qnet(test_data)
print(f"Output 0: {result.shape}")

test_conf = io_layer.fetch_config("1")
qnet = QNet.QNet(test_conf, all_conf)
print(f"Input 1: {test_data.shape}")
result = qnet(test_data)
print(f"Output 1: {result.shape}")

