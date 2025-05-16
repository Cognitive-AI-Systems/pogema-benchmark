import random

import numpy as np
import torch

from scrimp.alg_parameters import NetParameters, EnvParameters


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def reset_env(env, num_agent):
    done = False
    prev_action = np.zeros(num_agent)
    valid_actions = []
    obs = np.zeros((1, num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                   dtype=np.float32)
    vector = np.zeros((1, num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

    for i in range(num_agent):
        valid_action = env.list_next_valid_actions(i + 1)
        s = env.observe(i + 1)
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        vector[:, i, -1] = prev_action[i]
        valid_actions.append(valid_action)
        train_valid[i, valid_action] = 1
    return done, valid_actions, obs, vector, train_valid
