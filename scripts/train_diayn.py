import os
import yaml

from scripts.utils import make_env, get_latest_run_id
from stable_baselines.DIAYN.diayn import DIAYN
from stable_baselines.DIAYN.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

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

model = DIAYN.load("./log/diayn/HalfCheetah-v2_4/5200000/model.pkl", env=env,
                   tensorboard_log=tensorboard_log, verbose=1, save_path=save_path, **hyperparams)

# model = DIAYN(policy, env, tensorboard_log=tensorboard_log, verbose=1, save_path=save_path, **hyperparams)
model.learn(n_timesteps, seed=seed)

# saving
params_path = "{}/{}".format(save_path, env_id)
os.makedirs(save_path, exist_ok=True)
print("Saving to {}".format(save_path))
model.save(params_path)
