import QNet
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
    qnet = QNet.QNet(config, all_conf)
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
in_data = [test_data1, test_data2]

test_conf0 = io.fetch_config("0")
test_qnet(test_conf0, all_conf, in_data)

test_conf1 = io.fetch_config("1")
test_qnet(test_conf1, all_conf, in_data)

test_conf2 = io.fetch_config("2")
test_qnet(test_conf2, all_conf, in_data)

new_name = io.new_name()
print(f"\n*** Mating 0 and 2 to make {new_name} ***")
child_conf = mate.mate(test_conf0, test_conf2, new_name)
print(child_conf)
print(f"\n*** Mutating {new_name} ***")
mutate.mutate_candidate(child_conf, all_conf, 0.8, 0.8)

print(f"\n*** Correcting {new_name} ***")
correct.correct_candidate(child_conf, all_conf)
io.store_config(child_conf)
test_qnet(child_conf, all_conf, in_data)


