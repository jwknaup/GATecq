import datetime
import numpy as np
import random
import logging
import QNet

def mutate_layer(layer, all, expansion, mutation_rate):
    # FIXME: Not mutating
    return layer.copy()

def mutate_conv(conv, all, expansion, mutation_rate):
    # FIXME: Not mutating
    return conv.copy()

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
    layer_mutation_probability = 0.9 * mutation_rate / layer_count

    if 'lidar_conv' in candidate:
        # Should I mutate the lidar_conv?
        if random.random() < layer_mutation_probability:
            candidate['lidar_conv'] = mutate_conv(candidate['lidar_conv'], all, expansion, mutation_rate)
            logging.debug(f"Mutated lidar_conv")

    for i in range(len(candidate['layers'])):
        # Should I mutate this layer?
        if random.random() < layer_mutation_probability:
            candidate['layers'][i] = mutate_layer(candidate['layers'][i], all, expansion, mutation_rate)
            logging.debug(f"Mutated layer {i}")


        
        
        