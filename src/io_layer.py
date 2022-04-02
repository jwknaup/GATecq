import json

# This is designed to create an abstraction for reading and writing 
# the candidates and their stats

def fetch_all_config():
    with open("../initial_configs/_all.json") as f:
        result = json.load(f)
    return result

def fetch_config(config_name):
    with open(f"../initial_configs/{config_name}.conf") as f:
        result = json.load(f)
    return result

