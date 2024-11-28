import gymnasium as gym
import jax.numpy as jnp
import numpy as np

encoding = {' ': 0, ':': -1, '|': 1, 'G': -2, 'F': 2, 'P': 3, 'T': 5}
def encode(value):
    '''
    sum over the characters in the value encodings
    '''
    return sum([encoding[char] for char in value])

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

class MapWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        env.reset()

        self.observation_space = gym.spaces.Dict({
            'symbolic': env.observation_space,
            'domain_map': gym.spaces.Box(-15, 15, shape=self.prepare_domain_map().shape)
        })

    def prepare_domain_map(self):
        domain_map = get_domain_map(self.env)

        ret = list(map(lambda row: list(map(encode, row)), domain_map))
        ret = jnp.array(ret, dtype=jnp.float16)

        return ret

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)

        domain_map = self.prepare_domain_map()
        observation = jnp.array(observation, dtype=jnp.float16)
        observation = {
            'symbolic': observation,
            'domain_map': domain_map
        }
        return observation, {}
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)

        domain_map = self.prepare_domain_map()
        observation = jnp.array(observation, dtype=jnp.float16)
        observation = {
            'symbolic': observation,
            'domain_map': domain_map
        }

        return observation, reward, done, truncated, info

def eval_agent(env, agent):
    obs, _ = env.reset()
    done = False
    truncated = False

    total_reward = 0

    while not (done or truncated):
        action = agent(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    return total_reward
