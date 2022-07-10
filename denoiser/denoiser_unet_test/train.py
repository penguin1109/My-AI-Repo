## denoiser_unet_test/train.py ##

import os, json, copy, tqdm, shutil, argparse, time, warnings
import hydra, mlflow
import torch

import numpy as np

from code.aims_trainer import AIMSTrainer
from gli_aims.tools.mlops.aims_mlops import ManagedMLFlow
from config import Config
from jihye_trainer_entry import trainer_entry

def parse_args():
  parser = argparse.ArgumentParser(description = "Train the AIMS Denoiser")
  parser.add_argument('--config', help = 'FULL PATH of the Configuration file')
  parser.add_argument('--work-dir', help = 'Directory to save the logs and models', default = None)
  parser.add_argument('--phase', help = 'Number of the Training Phase', default = 1)
  args = parser.parse_args()

  return args

def main():
  args = parse_args()

  cfg_dir = args['config']
  config_instance = Config(cfg_dir)
  trainer_entry = trainer_entry(config_instance)
  my_app = hydra.main(config_path = "conf", 
                   config_name = "config")(trainer_entry.__call__)
  my_app()




