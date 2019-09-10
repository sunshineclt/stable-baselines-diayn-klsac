import os
import yaml
import gym
import numpy as np

from scripts.utils import create_test_env, get_latest_run_id, save_video
from stable_baselines.DIAYN.diayn import DIAYN
from stable_baselines.DIAYN.policies import MlpPolicy

algo = "diayn"
env_id = "HalfCheetah-v2"
seed = 0
deterministic = True
tensorboard_log = "log/tb/%s/" % env_id
n_timesteps = 1000
num_skills = 20
with open('diayn.yml', 'r') as f:
    hyperparams_dict = yaml.load(f)
    hyperparams = hyperparams_dict[env_id]

del hyperparams["n_timesteps"]
del hyperparams["policy"]
policy = MlpPolicy

log_folder = "log"
# log_path e.g. ./log/diayn/
log_path = "{}/{}/".format(log_folder, algo)
# save_path e.g. /log/diayn/HalfCheetah-v2_1
save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id)))
openai_log_path = os.path.join(save_path, "openai")
video_path = os.path.join(save_path, "videos")

env = create_test_env(env_id, n_envs=1,
                      seed=seed, log_dir=openai_log_path,
                      should_render=True, )

model = DIAYN.load("./log/diayn/HalfCheetah-v2_4/4600000/model.pkl", env=env,
                   tensorboard_log=None, verbose=1, **hyperparams)

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
