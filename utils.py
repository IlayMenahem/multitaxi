import collections
import random
import os

import jax.numpy as jnp
import jax
import rlax
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp

checkpoint_dir = 'checkpoint'
encoding = {' ': 0, ':': -1, '|': 1, 'G': -2, 'F': 2, 'P': 3, 'T': 5}

def encode(value):
    return sum(encoding[char] for char in value)

def get_passenger_locations(env):
    passengers = env.unwrapped.state().passengers
    passengers_locations = []

    for passenger in passengers:
        if not passenger.in_taxi:
            passengers_locations.append(passenger.location)

    return passengers_locations

def get_taxi_location(env):
    return env.unwrapped.state().taxis[0].location

def append_at_location(location, symbol, domain_map):
    row, col = location
    col *= 2
    domain_map[row, col] += symbol

    return domain_map

def get_domain_map(env):
    domain_map = env.unwrapped.domain_map.domain_map
    domain_map = np.array(domain_map[1:-1, 1:-1])

    for passenger_location in get_passenger_locations(env):
        domain_map = append_at_location(passenger_location, 'P', domain_map)
    domain_map = append_at_location(get_taxi_location(env), 'T', domain_map)

    return domain_map

def prepare_domain_map(env):
    domain_map = get_domain_map(env)

    ret = list(map(lambda row: list(map(encode, row)), domain_map))
    ret = jnp.array(ret, dtype=jnp.float16)

    return ret

def map_observation(env, observation):
    domain_map = prepare_domain_map(env)
    observation = jnp.array(observation, dtype=jnp.float16)

    return observation, domain_map

def get_shapes(env):
    domain_map_shape = prepare_domain_map(env).shape
    observation_shape = env.observation_space.shape[0]

    return observation_shape, domain_map_shape

def map_preprocess(env, obs):
    symbolic_obs, domain_map = map_observation(env, obs)

    obs = {'symbolic': symbolic_obs, 'domain_map': domain_map}
    return obs

def eval_agent_episode(env, agent, max_steps=50):
    obs, _ = env.reset()
    done = False
    truncated = False

    total_reward = 0

    for _ in range(max_steps):
        action = agent(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if done or truncated:
            break

    return total_reward

def eval_agent(env, agent, num_episodes=10):
    rewards = [eval_agent_episode(env, agent) for _ in range(num_episodes)]
    return rewards

class ReplayBuffer(object):
    """A simple Python replay buffer."""
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return tuple(zip(*batch))

    def is_ready(self):
        return len(self.buffer) >= self.batch_size

def epsilon_greedy(epsilon, qVals, episode, evaluation):
        train_a = rlax.epsilon_greedy(epsilon).sample(episode, qVals)
        eval_a = rlax.greedy().sample(episode, qVals)
        a = jax.lax.select(evaluation, eval_a, train_a)
        a = int(a[0])

def save_model(model: nnx.Module, chkp_dir: str, model_name: str) -> None:
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)
    model_path = os.path.join(os.path.abspath(chkp_dir), model_name)

    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(model_path, state, force=True)

def load_model(model: nnx.Module, chkp_dir: str, model_name: str) -> nnx.Module:
    model_path = os.path.join(os.path.abspath(chkp_dir), model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found')

    graphdf, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    resotred_state = checkpointer.restore(model_path, state)

    model = nnx.merge(graphdf, resotred_state)

    return model
