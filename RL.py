from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from multi_taxi import single_taxi_v0, maps
from utils import MapWrapper

'''
preformence by map
- small_no_obs_map: reward 225, 35 steps, 2 million training steps
- small_map: reward 280, 20 steps, 6.5 million training steps
- default_map: reward -, - steps, - million training steps
'''

env_kwargs = {'num_passengers':3,
    'max_fuel':50,
    'max_steps':250,
    'has_standby_action':True,
    'pickup_only':True,
    'observation_type':'symbolic',
    'domain_map':maps.DEFAULT_MAP,
    'render_mode':'human'
}

saves_path = "saves"
agent_name = "default_ppo"
eval_freq = int(5e5)
total_timesteps = int(2e7)

env_val = single_taxi_v0.gym_env(**env_kwargs)
env_val = MapWrapper(env_val)
env_val = Monitor(env_val, saves_path)

checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=saves_path, name_prefix=agent_name)
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5)
eval_callback = EvalCallback(env_val, eval_freq=eval_freq,
                callback_after_eval=stop_train_callback, verbose=0,
                deterministic=True, log_path=saves_path)
callbacks = [eval_callback, checkpoint_callback]

if __name__ == "__main__":
    env = single_taxi_v0.gym_env(**env_kwargs)
    env = MapWrapper(env)
    env = Monitor(env, saves_path)

    agent = PPO.load("saves/default_ppo.zip", env=env)
    agent.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
