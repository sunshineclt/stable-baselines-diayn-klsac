import os
import yaml

from scripts.utils import *
from stable_baselines.DIAYN.diayn_ppo import DIAYN_PPO
from stable_baselines.DIAYN.policies_ppo import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


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


algo = "diayn"
env_id = "CartPole-v1"
seed = 1
num_skills = 10
scale_intrinsic = 0.5
tensorboard_log = create_tensorboard_log_dir(env_id)
hyperparams = load_hyperparameter_from_yml("diayn_ppo.yml", env_id)
n_timesteps = int(hyperparams["n_timesteps"])
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

save_path = create_save_path("log", algo, env_id)

env = DummyVecEnv([make_env(env_id, 0, seed)], num_skills=num_skills)
policy = MlpPolicy

# Load and resume training
# model = DIAYN.load("./log/diayn/HalfCheetah-v2_4/5200000/model.pkl", env=env,
#                    tensorboard_log=tensorboard_log, verbose=1, save_path=save_path,
#                    **hyperparams)
# Train from scratch
model = DIAYN_PPO(policy, env, scale_intrinsic=scale_intrinsic,
                  tensorboard_log=tensorboard_log, verbose=1, save_path=save_path, num_skills=num_skills,
                  policy_kwargs={"num_skills": num_skills},
                  **hyperparams)

model.learn(n_timesteps, seed=seed)

# saving
params_path = "{}/final/model.pkl".format(save_path)
os.makedirs(os.path.join(save_path, "final"), exist_ok=True)
print("Saving to {}".format(save_path))
model.save(params_path)
