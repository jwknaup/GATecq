import datetime
import numpy as np
import random

def float_between(a, b, key):
    min_value = min(a[key], b[key])
    max_value = max(a[key], b[key])
    return np.random.uniform(min_value, max_value)

def int_between(a, b, key):
    min_value = min(a[key], b[key])
    max_value = max(a[key], b[key])
    return np.random.randint(min_value, max_value)

def mix_layers(a, b):
    # FIXME: I just pick one or the other
    if random.random() < 0.5:
        return a
    else:
        return b

def mix_lidar_convs(a, b):
    # FIXME: I just pick one or the other
    if random.random() < 0.5:
        return a
    else:
        return b

def mate(a, b, new_name):
    result = {}
    result['name'] = new_name
    result['parents'] = [a['name'], b['name']]
    result['creation_time'] = datetime.datetime.utcnow().isoformat(timespec='seconds')
    result['learning_rate'] = float_between(a, b, 'learning_rate')
    result['epsilon'] = float_between(a, b, 'epsilon')

    if 'lidar_conv' in a and 'lidar_conv' in b:
        result['lidar_conv'] = mix_lidar_convs(a['lidar_conv'], b['lidar_conv']);
    elif 'lidar_conv' in a: 
        if random.random() < 0.5:
            result['lidar_conv'] = a['lidar_conv']
    elif 'lidar_conv' in b:
        if random.random() < 0.5:
            result['lidar_conv'] = b['lidar_conv']

    a_layers = a['layers']
    b_layers = b['layers']
    min_layer_count = min(len(a_layers), len(b_layers))
    max_layer_count = max(len(a_layers), len(b_layers))
    layer_count = np.random.randint(min_layer_count, max_layer_count)
    result_layers = []
    for i in range(layer_count):
        if i < min_layer_count:
            result_layers.append(mix_layers(a_layers[i], b_layers[i]))
        else:
            if i < len(a_layers):
                result_layers.append(a_layers[i])
            else:
                result_layers.append(b_layers[i])
    result['layers'] = result_layers
    return result
