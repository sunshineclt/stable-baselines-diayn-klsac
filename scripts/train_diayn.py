import os
import yaml

from scripts.utils import *
from stable_baselines.DIAYN.diayn import DIAYN
from stable_baselines.DIAYN.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

algo = "diayn"
env_id = "Hopper-v3"
seed = 1
tensorboard_log = create_tensorboard_log_dir(env_id)
hyperparams = load_hyperparameter_from_yml("diayn.yml", env_id)
n_timesteps = int(hyperparams["n_timesteps"])
del hyperparams["n_timesteps"]
del hyperparams["policy"]
save_path = create_save_path("log", algo, env_id)

env = DummyVecEnv([make_env(env_id, 0, seed)])
policy = MlpPolicy

# Load and resume training
# model = DIAYN.load("./log/diayn/HalfCheetah-v2_4/5200000/model.pkl", env=env,
#                    tensorboard_log=tensorboard_log, verbose=1, save_path=save_path,
#                    **hyperparams)
# Train from scratch
model = DIAYN(policy, env,
              tensorboard_log=tensorboard_log, verbose=1, save_path=save_path,
              **hyperparams)

model.learn(n_timesteps, seed=seed)

# saving
params_path = "{}/final/model.pkl".format(save_path)
os.makedirs(save_path, exist_ok=True)
print("Saving to {}".format(save_path))
model.save(params_path)
