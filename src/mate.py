import datetime
import numpy as np
import random

# Utility functions
def float_between(a, b, key):
    min_value = min(a[key], b[key])
    max_value = max(a[key], b[key])
    return np.random.uniform(min_value, max_value)

def int_between(a, b, key):
    min_value = min(a[key], b[key])
    max_value = max(a[key], b[key])
    return np.random.randint(min_value, max_value)

# Merges layers
def mix_layers(a, b):
    # If not the same type, just pick one
    if  a['type'] != b['type']:
        if random.random() < 0.5:
            return a.copy()
        else:
            return b.copy()
    
    # a and b are the same type
    result = {'type':a['type']}
    result['output_dim'] = int_between(a, b, 'output_dim')

    if result['type'] == 'linear':
        activation = a['activation'];
        if random.random() < 0.5:
            activation = b['activation']
        result['activation'] = activation  

    if 'dropout' in a and 'dropout' in b:
        result['dropout'] = float_between(a, b, 'dropout')
    else:
        # Should we just copy a?
        if random.random() < 0.5:
            if 'dropout' in a:
                result['dropout'] = a['dropout']
        else:
            if 'dropout' in b:
                result['dropout'] = b['dropout']
    return result

# Merges lidar convolution layers
def mix_lidar_convs(a, b):
    result = {}
    result['kernel_size'] = int_between(a, b, 'kernel_size')

    stride = int_between(a, b, 'stride')
    output_channels = int_between(a, b, 'output_channels')
    # We are trying to *reduce* the dimension of the data
    if output_channels >= stride:
        # Nudge both
        output_channels = stride - 1
        stride = stride + 1
    result['stride'] = stride
    result['output_channels'] = output_channels

    if random.random() < 0.5:
        result['activation'] = a['activation']
    else:
        result['activation'] = b['activation']

    if 'dropout' in a and 'dropout' in b:
        result['dropout'] = float_between(a, b, 'dropout')
    else:
        # Should we just copy a?
        if random.random() < 0.5:
            if 'dropout' in a:
                result['dropout'] = a['dropout']
        else:
            if 'dropout' in b:
                result['dropout'] = b['dropout']

    return result

# Main function takes configuration for two systems and returns
# a new configuration that is a mix of the two
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
            result['lidar_conv'] = a['lidar_conv'].copy()
    elif 'lidar_conv' in b:
        if random.random() < 0.5:
            result['lidar_conv'] = b['lidar_conv'].copy()

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
