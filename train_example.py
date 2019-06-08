from arguments import get_args
from ppo_agent import ppo_agent
from models import cnn_net
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
import os

# create the environment
def create_single_football_env(args):
    """Creates gfootball environment."""
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True, with_checkpoints=False, 
            )
    return env

if __name__ == '__main__': 
    # get the arguments
    args = get_args()
    # create environments
    envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)], context=None)
    # create networks
    network = cnn_net(envs.action_space.n)
    # create the ppo agent
    ppo_trainer = ppo_agent(envs, args, network)
    ppo_trainer.learn()
