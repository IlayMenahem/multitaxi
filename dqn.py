"""A simple double-DQN agent trained to play BSuite's Catch env."""

from absl import flags
import sys

from multi_taxi import single_taxi_v0, maps

from utils import MapWrapper, ReplayBuffer
from agents import DQN

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("episode_count", 1000, "Number of episodes to train for.")
flags.DEFINE_integer("train_episodes", 301, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags.DEFINE_float("target_period", 50, "How often to update the target net.")
flags.DEFINE_integer("replay_capacity", 2000, "Capacity of the replay buffer.")
flags.DEFINE_float("epsilon_begin", 1., "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 1000, "Steps over which to anneal eps.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("evaluate_every", 50,
                     "Number of episodes between evaluations.")


def run_episode(env, agent, replay_buffer, episode, step_limit=250):
    obs_prev, _ = env.reset()
    epsidoic_reward = 0

    for step in range(step_limit):
        action = agent(obs_prev, episode, episode)
        obs, reward, done, truncated, _ = env.step(action)
        epsidoic_reward = reward + epsidoic_reward*FLAGS.discount_factor

        replay_buffer.push((obs_prev, action, epsidoic_reward, FLAGS.discount_factor, obs))
        obs_prev = obs

        if replay_buffer.is_ready() and step % FLAGS.target_period == 0:
            batch = replay_buffer.sample()
            agent.learner_step(*batch)

        if done or truncated:
            break

    print(f"reward: {epsidoic_reward:.3f}")

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
