import os
import yaml
import gym
import numpy as np
from tqdm import tqdm

from scripts.utils import *
from stable_baselines.DIAYN.diayn import DIAYN
from stable_baselines.DIAYN.policies import MlpPolicy


def evaluate(skill, repeat=0):
    obs = env.reset()
    obs = model.concat_obs_z(obs[0], skill)
    episode_reward = 0.0
    imgs = []
    for timestep in range(n_timesteps):
        action = model.predict(obs[None], deterministic=deterministic)[0]  # 0 is action, 1 is logp
        # Random Agent
        # action = [env.action_space.sample()]
        obs, reward, done, infos = env.step(action)
        obs = model.concat_obs_z(obs[0], skill)
        imgs.append(env.render('rgb_array'))

        episode_reward += reward[0]

        if done:
            assert timestep == n_timesteps - 1, "Done before assigned timestep! "

    tqdm.write("Skill {}, Repeat {}, Episode Reward: {:.2f}".format(skill, repeat, episode_reward))
    save_video(imgs, os.path.join(video_path, "skill_%d_repeat_%d.avi" % (skill, repeat)))


if __name__ == "__main__":
    # Parameters! Take a good look before execution!!!
    algo = "diayn"
    env_id = "InvertedPendulum-v2"
    trained_iter_number = "final"
    seed = 0
    deterministic = True
    tensorboard_log = create_tensorboard_log_dir(env_id)
    n_timesteps = 1000
    num_skills = 20
    hyperparams = load_hyperparameter_from_yml("diayn.yml", env_id)
    del hyperparams["n_timesteps"]
    del hyperparams["policy"]
    hyperparams["num_skills"] = num_skills

    # Create paths
    save_path = create_save_path("log", algo, env_id, id=1)
    openai_log_path = os.path.join(save_path, "openai_" + trained_iter_number)
    video_path = os.path.join(save_path, "videos_" + trained_iter_number)
    model_path = os.path.join(save_path, trained_iter_number, "model.pkl") \
        if trained_iter_number != "final" \
        else os.path.join(save_path, env_id + ".pkl")

    # Load model
    # NOTE: env created will be vectorized (though n_envs=1)
    env = create_test_env(env_id, n_envs=1, seed=seed, log_dir=openai_log_path, should_render=True, )
    # not general
    policy = MlpPolicy
    model = DIAYN.load(model_path,
                       env=env, tensorboard_log=None, verbose=1, **hyperparams)

    for skill in tqdm(range(num_skills)):
        for repeat in range(3):
            evaluate(skill, repeat=repeat)
