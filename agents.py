from collections import deque
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import rlax
import gymnasium as gym
from multi_taxi import single_taxi_v0
from utils import get_taxi_location, get_passenger_locations
import numpy as np

from utils import map_observation, get_shapes
from models import MultiTaxi, probabilityMultiTaxi


def recounstruct_path(prev, location):
    path = []
    while location:
        path.append(location)
        location = prev[location]

    path.reverse()

    return path

def bfs_pathfinding(env, start, goals):
    """
    Finds the shortest path from start to goal in the domain_map using BFS.

    Parameters:
    - env: gym.Env object.
    - start: Tuple (row, col) indicating the starting position.
    - goal: Tuple (row, col) indicating the goal position.

    Returns:
    - path: List of tuples representing the path from start to goal.
            Returns None if no path is found.
    """
    domain_map = env.unwrapped.domain_map
    rows = domain_map.map_height
    cols = domain_map.map_width

    visited = np.zeros((rows, cols), dtype=bool)
    prev = np.full((rows, cols), None, dtype=object)

    queue = deque()
    queue.append(start)
    visited[start] = True

    while queue:
        current = queue.popleft()

        if current in goals:
            return recounstruct_path(prev, current)

        row, col = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                if not visited[r, c] and not domain_map.hit_obstacle(current, (r, c)):
                    visited[r, c] = True
                    prev[r, c] = current
                    queue.append((r, c))

    return None

class BfsAgent:
    def __init__(self, env: single_taxi_v0.gym_env):
        self.env = env

        action_to_name = env.unwrapped.get_action_map()
        self.action_map = {
            (1, 0): action_to_name['south'],
            (-1, 0): action_to_name['north'],
            (0, -1): action_to_name['west'],
            (0, 1): action_to_name['east'],
            'pickup': action_to_name['pickup'],
            'refuel': action_to_name['refuel'],
        }

    def __call__(self, obs):
        taxi_loc = get_taxi_location(self.env)
        passenger_locs = get_passenger_locations(self.env)

        path = bfs_pathfinding(self.env, taxi_loc, passenger_locs)
        # agent is at the passenger location, so return pickup action
        if len(path) == 1:
            return self.action_map['pickup']

        next_loc = path[1]
        direction = (next_loc[0] - taxi_loc[0], next_loc[1] - taxi_loc[1])
        action = self.action_map[direction]

        return action

def preprocess_batch(batch):
    '''
    convert the batch of observations to jax arrays
    :param batch: Tuple of lists containing the domain map and symbolic observation
    :return: Tuple of jax arrays
    '''
    symbolic_obs = jnp.array([x['symbolic'] for x in batch], dtype=jnp.float16)
    domain_map = jnp.array([x['domain_map'] for x in batch], dtype=jnp.float16)

    return symbolic_obs, domain_map

def preprocess(obs):
    '''
    convert the observation to jax arrays
    :param obs: Tuple of dictionaries containing the domain map and symbolic observation
    :return: Tuple of jax arrays
    '''
    domain_map = jnp.array(obs['domain_map'], dtype=jnp.float16)
    symbolic_obs = jnp.array(obs['symbolic'], dtype=jnp.float16)

    return symbolic_obs, domain_map

class MultiTaxiAgent(ABC):
    @abstractmethod
    def __init__(self, env, learning_rate):
        pass

    @abstractmethod
    def __call__(self, obs):
        pass

    @abstractmethod
    def learner_step(self):
        pass

class BCAgent(MultiTaxiAgent):
    def __init__(self, env: gym.Env, learning_rate=0.001):
        self.env = env
        symbolic_shape, img_shape = get_shapes(env)
        self.num_actions = env.action_space.n
        self.model = probabilityMultiTaxi(img_shape, symbolic_shape, self.num_actions)

        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))
        self._optimizer = nnx.Optimizer(self.model, optimizer)

    def __call__(self, obs, pruned_action=None):
        self.model.eval()

        symbolic_obs, domain_map = map_observation(self.env, obs)
        probs = self.model(symbolic_obs, domain_map)

        if pruned_action is not None:
            probs = jax.ops.index_update(probs, pruned_action, 0.0)
            probs = probs / jnp.sum(probs)

        a = jnp.argmax(probs, axis=-1)
        a = int(a[0])

        return a

    def learner_step(self, obs, expert_action):
        self.model.train()

        symbolic_obs, domain_map = preprocess_batch(obs)
        expert_action_one_hot = jax.nn.one_hot(expert_action, self.num_actions)

        def loss_fn(model, symbolic_obs, domain_map, expert_action):
            probs = model(symbolic_obs, domain_map)
            return jnp.sum((expert_action - probs)**2)

        grad_fn = nnx.value_and_grad(nnx.jit(loss_fn))
        loss, grad = grad_fn(self.model, symbolic_obs, domain_map, expert_action_one_hot)
        self._optimizer.update(grad)

        return loss

class DQN(MultiTaxiAgent):
    def __init__(self, env, epsilon_cfg, learning_rate):
        img_shape = env.observation_space['domain_map'].shape
        symbolic_shape = env.observation_space['symbolic'].shape[0]
        num_actions = env.action_space.n

        self.model = MultiTaxi(img_shape, symbolic_shape, num_actions)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)

        optimizer = optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(learning_rate))
        self._optimizer = nnx.Optimizer(self.model, optimizer)

    def __call__(self, obs, episode, key, evaluation=False):
        self.model.eval()

        symbolic_obs, domain_map = preprocess(obs)

        qVals = self.model(symbolic_obs, domain_map)
        epsilon = self._epsilon_by_frame(episode)

        train_a = rlax.epsilon_greedy(epsilon).sample(key, qVals)
        eval_a = rlax.greedy().sample(key, qVals)
        a = jax.lax.select(evaluation, eval_a, train_a)
        a = int(a[0])

        return a

    def learner_step(self, obs_tm1, a_tm1, r_t, discount_t, obs_t):
        self.model.train()

        obs_tm1 = preprocess_batch(obs_tm1)
        obs_t = preprocess_batch(obs_t)
        a_tm1 = jnp.array(a_tm1, dtype=jnp.int16)
        r_t = jnp.array(r_t, dtype=jnp.float16)
        discount_t = jnp.array(discount_t, dtype=jnp.float16)

        grad_fn = nnx.value_and_grad(self._loss)
        loss, grad = grad_fn(self.model, obs_tm1, a_tm1, r_t, discount_t, obs_t)
        self._optimizer.update(grad)

        return loss

    @staticmethod
    def _loss(model, obs_tm1, a_tm1, r_t, discount_t, obs_t):
        q_tm1 = model(*obs_tm1)
        q_t = model(*obs_t)

        td_error = jax.vmap(rlax.q_learning)(q_tm1, a_tm1, r_t, discount_t, q_t)
        loss = jnp.mean(rlax.l2_loss(td_error))

        return loss


