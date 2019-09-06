import os
import yaml
import time
import gym
import glob

from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.DIAYN.diayn import DIAYN
from stable_baselines.DIAYN.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                    env with
    """
    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


algo = "diayn"
env_id = "HalfCheetah-v2"
seed = 0
tensorboard_log = "log/tb/%s/" % env_id
with open('diayn.yml', 'r') as f:
    hyperparams_dict = yaml.load(f)
    hyperparams = hyperparams_dict[env_id]

n_timesteps = int(hyperparams["n_timesteps"])
del hyperparams["n_timesteps"]
del hyperparams["policy"]

env = DummyVecEnv([make_env(env_id, 0, seed)])
policy = MlpPolicy

log_folder = "log"
log_path = "{}/{}/".format(log_folder, algo)
save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))

model = DIAYN(policy, env, tensorboard_log=tensorboard_log, verbose=1, save_path=save_path, **hyperparams)
model.learn(n_timesteps)

# saving
params_path = "{}/{}".format(save_path, env_id)
os.makedirs(save_path, exist_ok=True)
print("Saving to {}".format(save_path))
model.save(params_path)
