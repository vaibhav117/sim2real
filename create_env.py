from ast import arg
import numpy as np
import gym
import gym_point
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import wandb
from datetime import datetime
from pytz import timezone
from MakeTreeDir import MAKETREEDIR
from typing import Dict, Iterable, Optional, Union
import glob
import pickle

args = get_args()

def create_env():
    env = gym.make(
                    args.env_name,
                    view=args.view,
                    reward_type=args.reward_type,
                    depth=args.depth, 
                    texture_rand=args.texture_rand, 
                    camera_rand=args.camera_rand, 
                    depth_noise=args.depth_noise, 
                    light_rand=args.light_rand,
                    crop_amount=args.crop_amount,
                    action_variability=args.action_variability
                )

    return env