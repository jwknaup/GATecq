import datetime
import numpy as np
import random
import logging
import QNet


def mutate_layer(layer, all, expansion, mutation_rate):
    # Start with an exact copy
    result = layer.copy()
    if result['type'] == 'linear':
        layer_change_activation_probability = mutation_rate
        if random.random() < layer_change_activation_probability:
            all_activations = QNet.all_activations()
            old_activation = result['activation']
            while result['activation'] == old_activation:
                result['activation'] = random.choice(all_activations)
            logging.debug(f"Mutated activation from {old_activation} to {result['activation']}")

    if 'dropout' in layer:
        layer_change_dropout_probability = mutation_rate
        if random.random() < layer_change_dropout_probability:
            old_value = result['dropout']
            # It can increase or decrease by up to 50%
            delta = old_value * random.random() - (old_value * 0.5)
            result['dropout'] = old_value + delta
            logging.debug(f"Mutated dropout to {result['dropout']}")

    layer_change_size_probability = mutation_rate
    if random.random() < layer_change_size_probability:
        old_value = result['output_dim']
        # It can increase or decrease by up to 50%
        delta = int(old_value * random.random() - old_value * 0.5 * expansion)
        result['output_dim'] = old_value + delta
        logging.debug(f"Mutated output_dim from {old_value} to {result['output_dim']}")

    layer_rnn_method_change = 0.3 * mutation_rate
    if random.random() < layer_rnn_method_change:
        if result['type'] == 'lstm':
            result['type'] = 'gru'
            logging.debug(f"Mutated type from lstm to gru")
        elif result['type'] == 'gru':
            result['type'] = 'lstm'
            logging.debug(f"Mutated type from gru to lstm")
    return result


def mutate_conv(conv, all, expansion, mutation_rate):
    result = conv.copy()
    layer_change_activation_probability = mutation_rate
    if random.random() < layer_change_activation_probability:
        all_activations = QNet.all_activations()
        old_activation = result['activation']
        while result['activation'] == old_activation:
            result['activation'] = random.choice(all_activations)
        logging.debug(f"Mutated activation from {old_activation} to {result['activation']}")

        stride_change_size_probability = mutation_rate
        if random.random() < stride_change_size_probability:
            old_value = result['stride']
            # It can increase or decrease by up to 50%
            delta = int(old_value * random.random() - old_value * 0.5/expansion)
            result['stride'] = old_value + delta
            logging.debug(f"Mutated stride from {old_value} to {result['stride']}")

        kernel_change_size_probability = mutation_rate
        if random.random() < kernel_change_size_probability:
            old_value = result['kernel_size']
            # It can increase or decrease by up to 50%
            delta = int(old_value * random.random() - old_value * 0.5/expansion)
            result['kernel_size'] = old_value + delta
            logging.debug(f"Mutated kernel_size from {old_value} to {result['kernel_size']}")

        output_channels_change_size_probability = mutation_rate
        if random.random() < output_channels_change_size_probability:
            old_value = result['output_channels']
            # It can increase or decrease by up to 50%
            delta = int(old_value * random.random() - old_value * 0.5 * expansion)
            result['output_channels'] = old_value + delta
            logging.debug(f"Mutated output_channels from {old_value} to {result['output_channels']}")

    return result


# mutate() takes a candidate configuration, the common data for all candidates,
# an expansion ( <1 means less complexity, >1 means more complexity), and
# a mutation rate (0-1) which indicates how big and common mutations are.
def mutate_candidate(candidate, all, expansion, mutation_rate):

    # Should I duplicate a layer?
    layer_duplication_probability = 0.3 * mutation_rate * expansion
    if random.random() < layer_duplication_probability:
        layer_to_duplicate = random.randint(0, len(candidate['layers']) - 1)
        new_layer = candidate['layers'][layer_to_duplicate].copy()
        candidate['layers'].insert(layer_to_duplicate, new_layer)
        logging.debug(f"Duplicated layer {layer_to_duplicate}")

    # Should I delete a layer?
    layer_removal_probability = 0.3 * mutation_rate / expansion
    if random.random() < layer_removal_probability:
        layer_to_delete = random.randint(0, len(candidate['layers']) - 1)
        candidate['layers'].pop(layer_to_delete)
        logging.debug(f"Deleted layer {layer_to_delete}")

    layer_count = len(candidate['layers'])
    layer_mutation_probability = 0.6 * mutation_rate 

    if 'lidar_conv' in candidate:
        # Should I mutate the lidar_conv?
        if random.random() < layer_mutation_probability:
            logging.debug(f"Mutating lidar_conv")
            candidate['lidar_conv'] = mutate_conv(candidate['lidar_conv'], all, expansion, mutation_rate)

    for i in range(len(candidate['layers'])):
        # Should I mutate this layer?
        if random.random() < layer_mutation_probability:
            logging.debug(f"Mutating layer {i}")
            candidate['layers'][i] = mutate_layer(candidate['layers'][i], all, expansion, mutation_rate)
