"""A simple double-DQN agent trained to play BSuite's Catch env."""
# fix reward counting
# implement the learner_step method

import collections
import sys
import random
from absl import flags
from multi_taxi import single_taxi_v0, maps

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import rlax

from models import MultiTaxi
from utils import MapWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("episode_count", 1000, "Number of episodes to train for.")
flags.DEFINE_integer("train_episodes", 301, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags.DEFINE_float("target_period", 50, "How often to update the target net.")
flags.DEFINE_integer("replay_capacity", 2000, "Capacity of the replay buffer.")
flags.DEFINE_integer("hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_float("epsilon_begin", 1., "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 1000, "Steps over which to anneal eps.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50,
                     "Number of episodes between evaluations.")

class ReplayBuffer(object):
    """A simple Python replay buffer."""
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, prev_obs, action, reward, discount, obs):
        data = (prev_obs, action, reward, discount, obs)
        self.buffer.append(data)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return tuple(zip(*batch))


    def is_ready(self):
        return len(self.buffer) >= self.batch_size


class DQN:
    def __init__(self, env, epsilon_cfg, learning_rate):
        img_shape = env.observation_space['domain_map'].shape
        symbolic_shape = env.observation_space['symbolic'].shape[0]
        num_actions = env.action_space.n

        self.model = MultiTaxi(img_shape, symbolic_shape, num_actions)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate)
        )
        self._optimizer = nnx.Optimizer(self.model, optimizer)
        self._metrics = nnx.metrics.Average('reward')

    def preprocess(self, obs):
        '''
        convert the observation to jax arrays
        :param obs: Tuple of dictionaries containing the domain map and symbolic observation
        :return: Tuple of jax arrays
        '''
        domain_map = jnp.array(obs['domain_map'], dtype=jnp.float16)
        symbolic_obs = jnp.array(obs['symbolic'], dtype=jnp.float16)

        return symbolic_obs, domain_map

    def preprocess_batch(self, batch):
        '''
        convert the batch of observations to jax arrays
        :param batch: Tuple of lists containing the domain map and symbolic observation
        :return: Tuple of jax arrays
        '''
        symbolic_obs = jnp.array([x['symbolic'] for x in batch], dtype=jnp.float16)
        domain_map = jnp.array([x['domain_map'] for x in batch], dtype=jnp.float16)

        return symbolic_obs, domain_map

    def actor_step(self, obs, episode, key, evaluation=False):
        self.model.eval()

        symbolic_obs, domain_map = self.preprocess(obs)

        qVals = self.model(symbolic_obs, domain_map)
        epsilon = self._epsilon_by_frame(episode)

        train_a = rlax.epsilon_greedy(epsilon).sample(key, qVals)
        eval_a = rlax.greedy().sample(key, qVals)
        a = jax.lax.select(evaluation, eval_a, train_a)
        a = int(a[0])

        return a

    #@nnx.jit
    def learner_step(self, obs_tm1, a_tm1, r_t, discount_t, obs_t):
        self.model.train()

        obs_tm1 = self.preprocess_batch(obs_tm1)
        obs_t = self.preprocess_batch(obs_t)
        a_tm1 = jnp.array(a_tm1, dtype=jnp.int16).reshape(-1, 1)
        r_t = jnp.array(r_t, dtype=jnp.float16).reshape(-1, 1)
        discount_t = jnp.array(discount_t, dtype=jnp.float16).reshape(-1, 1)

        grad_fn = nnx.value_and_grad(self._loss)
        loss, grad = grad_fn(self.model, obs_tm1, a_tm1, r_t, discount_t, obs_t)
        self._optimizer.update(grad)

        return loss

    #@nnx.jit
    @staticmethod
    def _loss(model, obs_tm1, a_tm1, r_t, discount_t, obs_t):
        q_tm1 = model(*obs_tm1)
        q_t_val = model(*obs_t)
        q_t_select = q_t_val

        td_error = nnx.vmap(rlax.double_q_learning)(q_tm1, a_tm1, r_t, discount_t,
                                                     q_t_val, q_t_select)
        raise NotImplementedError("Implement the loss function")
        loss = jnp.mean(rlax.l2_loss(td_error))

        return loss

def run_episode(env, agent, replay_buffer, episode, step_limit=250):
    obs_prev, _ = env.reset()

    for step in range(step_limit):
        action = agent.actor_step(obs_prev, episode, jax.random.PRNGKey(episode))
        obs, reward, done, truncated, _ = env.step(action)

        replay_buffer.push(obs_prev, action, reward, FLAGS.discount_factor, obs)
        obs_prev = obs

        if replay_buffer.is_ready() and step % FLAGS.target_period == 0:
            batch = replay_buffer.sample()
            agent.learner_step(*batch)

        if done or truncated:
            break

def train_loop(episode_count):
    env = single_taxi_v0.gym_env(
        num_passengers=3,
        max_fuel=50,
        max_steps=250,
        has_standby_action=True,
        pickup_only=True,
        observation_type='symbolic',
        domain_map=maps.DEFAULT_MAP,
        render_mode='human'
    )
    env.seed(FLAGS.seed)
    env = MapWrapper(env)

    epsilon_cfg = dict(init_value=FLAGS.epsilon_begin, end_value=FLAGS.epsilon_end,
        transition_steps=FLAGS.epsilon_steps, power=1.)
    agent = DQN(env, epsilon_cfg, FLAGS.learning_rate)

    replay_buffer = ReplayBuffer(FLAGS.replay_capacity, FLAGS.batch_size)

    for episode in range(episode_count):
        run_episode(env, agent, replay_buffer, episode)

if __name__ == "__main__":
    FLAGS(sys.argv)
    train_loop(FLAGS.episode_count)
