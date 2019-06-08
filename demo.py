from arguments import get_args
from ppo_agent import ppo_agent
import gfootball.env as football_env
import numpy as np
from models import cnn_net
import torch
import os

# get the tensors
def get_tensors(obs):
    return torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)

if __name__ == '__main__':
    args = get_args()
    model_path = args.save_dir + args.env_name + '/model.pt'
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True, with_checkpoints=False,
            enable_full_episode_videos=False, render=True)
    network = cnn_net(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # start to do the test
    obs = env.reset()
    for _ in range(100):
        obs_tensor = get_tensors(np.expand_dims(obs, 0))
        with torch.no_grad():
            _, pi = network(obs_tensor)
        actions = torch.argmax(pi, dim=1).item()
        obs, reward, done, _ = env.step(actions)
        if done:
            obs = env.reset()
    env.close()
