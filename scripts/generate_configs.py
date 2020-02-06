# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:49:02 2020

@author: Nitin

simulation.cycles 500

"""

## File to create config files automatically for the respective datasets

import pandas as pd
import os
import numpy as np
from copy import copy
from collections import Counter
import logging
import random
import math



# constants
DATA_DIR = "../data/"

def load_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_gen_file", type=str,
                        help="path of config gen file")
    args = parser.parse_args()
    return args


def load_base_config(path):
    with open(path, "r") as f:
        base_config_lines = f.readlines()
        return base_config_lines
    return []


def append_to_base_config(base_config_lines, config):
    for key in config.keys():
        base_config_lines.append(key +" " + config[key])
    return base_config_lines
    
    
def load_config_gen_file(path):
    from collections import defaultdict
    config_dict = defaultdict(dict)
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            config_values = line.lstrip().rstrip().split(",")
            assert len(config_values) == 13
            
            
            config_dict["config_{}".format(i)]["simulation.cycles"] = config_values[0]
            config_dict["config_{}".format(i)]["network.size"] = config_values[1]
            config_dict["config_{}".format(i)]["network.node.resourcepath"] = config_values[2]
            config_dict["config_{}".format(i)]["network.node.trainlen"] = config_values[3]
            config_dict["config_{}".format(i)]["network.node.testlen"] = config_values[4]
            
            config_dict["config_{}".format(i)]["network.node.numhidden"] = config_values[5]
            config_dict["config_{}".format(i)]["network.node.learningrate"] = config_values[6]
            config_dict["config_{}".format(i)]["network.node.cyclesforconvergence"] = config_values[7]
            config_dict["config_{}".format(i)]["network.node.convergenceepsilon"] = config_values[8]
            config_dict["config_{}".format(i)]["network.node.lossfunction"] = config_values[9]
            config_dict["config_{}".format(i)]["network.node.hidden_layer_act"] = config_values[10]
            config_dict["config_{}".format(i)]["network.node.final_layer_act"] = config_values[11]
            
            # needed just for the sake of the code
            config_dict["config_{}".format(i)]["dataset"] = config_values[12]
            
    return config_dict


if __name__ == "__main__":
    args = load_args()
    config_dict = load_config_gen_file(args.config_gen_file)
    
    # Create a file for each of these
    for config in config_dict.keys():
        base_config_lines = load_base_config("../config/base_config.cfg")
        updated_config = append_to_base_config(base_config_lines, config_dict[config])
        
        op_dir = os.path.join("../config", config_dict[config]["dataset"])
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        
        with open(os.path.join(op_dir, config + ".cfg"), "w") as f:  
            f.write("\n".join(updated_config))
        

