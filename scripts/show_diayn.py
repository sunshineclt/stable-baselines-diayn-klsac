import os
import yaml
import gym
import numpy as np

from scripts.utils import *
from stable_baselines.DIAYN.diayn import DIAYN
from stable_baselines.DIAYN.policies import MlpPolicy

algo = "diayn"
env_id = "HalfCheetah-v2"
trained_iter_number = "1400000"
seed = 0
deterministic = True
tensorboard_log = create_tensorboard_log_dir(env_id)
n_timesteps = 1000
num_skills = 20
hyperparams = load_hyperparameter_from_yml("diayn.yml", env_id)
del hyperparams["n_timesteps"]
del hyperparams["policy"]
hyperparams["num_skills"] = num_skills

save_path = create_save_path("log", algo, env_id, id=5)
openai_log_path = os.path.join(save_path, "openai_" + trained_iter_number)
video_path = os.path.join(save_path, "videos_" + trained_iter_number)

env = create_test_env(env_id, n_envs=1, seed=seed, log_dir=openai_log_path, should_render=True, )
policy = MlpPolicy

model = DIAYN.load(os.path.join(save_path, trained_iter_number, "/model.pkl"),
                   env=env, tensorboard_log=None, verbose=1, **hyperparams)

for skill in range(num_skills):
    obs = env.reset()
    obs = model.concat_obs_z(obs[0], skill)
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    done = False
    imgs = []
    print("skill", skill)
    for _ in range(n_timesteps):
        action = model.predict(obs[None], deterministic=deterministic)
        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        obs = model.concat_obs_z(obs[0], skill)
        img = env.render('rgb_array')
        imgs.append(img)

        episode_reward += reward[0]
        ep_len += 1

        if done:
            # NOTE: for env using VecNormalize, the mean reward
            # is a normalized reward when `--norm_reward` flag is passed
            print("Episode Reward: {:.2f}".format(episode_reward))
            print("Episode Length", ep_len)
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            ep_len = 0

    save_video(imgs, os.path.join(video_path, "skill_%d.avi" % (skill, )))

    if len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))
