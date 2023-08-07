import os
import pickle

import yaml
import gym
import numpy as np
from tqdm import tqdm

from scripts.utils import *
from stable_baselines.DIAYN.diayn_ppo import DIAYN_PPO
from stable_baselines.DIAYN.policies_ppo import MlpPolicy


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def evaluate(skill, repeat=0):
    obs = env.reset()
    obs = model.concat_obs_z(obs[0, :-10], skill)
    episode_reward = 0.0
    imgs = []
    ob_saved = []
    act_saved = []
    for timestep in range(n_timesteps):
        action = model.predict(obs[None], deterministic=deterministic)[0]  # 0 is action, 1 is logp
        ob_saved.append(obs[:4])
        act_saved.append(action[0])
        # Random Agent
        # action = [env.action_space.sample()]
        obs, reward, done, infos = env.step(action)
        obs = model.concat_obs_z(obs[0, :-10], skill)
        imgs.append(env.render('rgb_array'))

        episode_reward += reward[0]

        if done:
            # break
            assert timestep == n_timesteps - 1, "Done before assigned timestep! "

    with open(f"data_saved_for_droid_3/{skill}@{repeat}-.pkl", "wb") as f:
        pickle.dump({"states": np.array(ob_saved), "actions": np.array(act_saved)}, f)
    tqdm.write("Skill {}, Repeat {}, Episode Reward: {:.2f}".format(skill, repeat, episode_reward))
    save_video(imgs, os.path.join(video_path, "skill_%d_repeat_%d.avi" % (skill, repeat)), fps=60.0)


if __name__ == "__main__":
    # Parameters! Take a good look before execution!!!
    algo = "diayn"
    env_id = "LunarLander-v2"
    trained_iter_number = "final"
    seed = 0
    deterministic = True
    tensorboard_log = create_tensorboard_log_dir(env_id)
    n_timesteps = 200
    num_skills = 20
    hyperparams = load_hyperparameter_from_yml("diayn_ppo.yml", env_id)
    del hyperparams["n_timesteps"]
    del hyperparams["policy"]
    del hyperparams["n_envs"]
    for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = float(hyperparams[key])
        else:
            raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))
    hyperparams["num_skills"] = num_skills

    # Create paths
    save_path = create_save_path("log", algo, env_id, id=5)
    openai_log_path = os.path.join(save_path, "openai_" + trained_iter_number)
    video_path = os.path.join(save_path, "videos_" + trained_iter_number)
    model_path = os.path.join(save_path, trained_iter_number, "model.pkl") \
        if trained_iter_number == "final" \
        else os.path.join(save_path, env_id + ".pkl")

    # Load model
    # NOTE: env created will be vectorized (though n_envs=1)
    env = create_test_env(env_id, n_envs=1, seed=seed, log_dir=openai_log_path, should_render=True, )
    env.close()
    # not general
    policy = MlpPolicy
    model = DIAYN_PPO.load(model_path,
                           env=env, tensorboard_log=None, verbose=1, **hyperparams)

    all_traj_name_skill_maps = {}
    for skill in tqdm(range(num_skills)):
        for repeat in range(3):
            env = create_test_env(env_id, n_envs=1, seed=repeat, log_dir=openai_log_path, should_render=True, )
            evaluate(skill, repeat=repeat)
            env.envs[0].unwrapped.close()
            env.close()
            del env
            all_traj_name_skill_maps[f"{skill}@{repeat}"] = skill
    import json
    with open("data_saved_for_droid_3/traj_matched_with_strategy.json", "w") as f:
        json.dump(all_traj_name_skill_maps, f)

