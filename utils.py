import numpy as np
import torch
from torch.distributions.categorical import Categorical
import random

# select actions
def select_actions(pi):
    actions = Categorical(pi).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()

# evaluate actions
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
    entropy = cate_dist.entropy().mean()
    return log_prob, entropy
