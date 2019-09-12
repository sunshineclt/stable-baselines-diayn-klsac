import glob
import os
import time

import gym
import yaml

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv


def create_test_env(env_id, n_envs=1, seed=0,
                    log_dir='', should_render=True):
    """
    Create environment for testing a trained agent
    NOTE: the env will be vectorized!!!

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param env_wrapper: (type) A subclass of gym.Wrapper to wrap the original
                        env with
    :return: (gym.Env)
    """
    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = 'csv'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=None)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    return env


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


def save_video(ims, filename):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


def create_tensorboard_log_dir(env_id):
    return "log/tb/%s/" % env_id


def load_hyperparameter_from_yml(filename, env_id):
    with open(filename, 'r') as f:
        hyperparams_dict = yaml.load(f)
        hyperparams = hyperparams_dict[env_id]
    return hyperparams


def create_save_path(log_root, algo, env_id, id=None):
    log_path = "{}/{}/".format(log_root, algo)
    if id:
        save_path = os.path.join(log_path, "{}_{}".format(env_id, id))
    else:
        save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
    os.makedirs(save_path, exist_ok=True)
    return save_path
