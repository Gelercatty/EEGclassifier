from types import SimpleNamespace as Namespace
import yaml
import torch
import yaml
import argparse
import random
import numpy as np
def load_config(file):
    """ 加载配置文件并返回一个命名空间对象 """
    with open(file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return recursive_namespace(data_loaded)

def recursive_namespace(d):
    """ 递归地将字典转换为命名空间 """
    if isinstance(d, dict):
        return Namespace(**{k: recursive_namespace(v) for k, v in d.items()})
    return d

def parse_args():
    """ 解析命令行参数并加载配置 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = load_config(args.config_file)
    return config


