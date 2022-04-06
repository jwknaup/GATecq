import QNet
import io_layer
import numpy as np 
import torch
import mate
import logging

logging.basicConfig(level=logging.DEBUG)

io = io_layer.io_layer()

all_conf = io.fetch_all_config()

print("Making 2 test inputs")
test_data1 = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))
test_data2 = torch.Tensor(np.random.rand(1, all_conf["other_inputs"] + all_conf["lidar_inputs"]))

print("\n*** Candidate 0 ***")
test_conf0 = io.fetch_config("0")
print(test_conf0)
qnet = QNet.QNet(test_conf0, all_conf)
print(f"Input: {test_data1.shape}")
result = qnet(test_data1)
print(f"Output 0: {result.shape}")
result = qnet(test_data2)
print(f"Output 1: {result.shape}")

print("\n*** Candidate 1 ***")
test_conf1 = io.fetch_config("1")
print(test_conf1)
qnet = QNet.QNet(test_conf1, all_conf)
print(f"Input: {test_data1.shape}")
result = qnet(test_data1)
print(f"Output 0: {result.shape}")
result = qnet(test_data2)
print(f"Output 1: {result.shape}")

print("\n*** Candidate 2 ***")
test_conf2 = io.fetch_config("2")
print(test_conf2)
qnet = QNet.QNet(test_conf2, all_conf)
print(f"Input: {test_data1.shape}")
result = qnet(test_data1)
print(f"Output 0: {result.shape}")
result = qnet(test_data2)
print(f"Output 1: {result.shape}")

new_name = io.new_name()
child_conf = mate.mate(test_conf0, test_conf1, new_name)
print(f"\n*** Candidate {new_name} ***\n{child_conf}")
io.store_config(child_conf)
qnet = QNet.QNet(child_conf, all_conf)
print(f"Input: {test_data1.shape}")
result = qnet(test_data1)
print(f"Output 0: {result.shape}")
result = qnet(test_data2)
print(f"Output 1: {result.shape}")

new_name = io.new_name()
child_conf = mate.mate(test_conf0, test_conf2, new_name)
print(f"\n*** Candidate {new_name} ***\n{child_conf}")
io.store_config(child_conf)
qnet = QNet.QNet(child_conf, all_conf)
print(f"Input: {test_data1.shape}")
result = qnet(test_data1)
print(f"Output 0: {result.shape}")
result = qnet(test_data2)
print(f"Output 1: {result.shape}")

