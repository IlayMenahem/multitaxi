"""A simple double-DQN agent trained to play BSuite's Catch env."""

from absl import flags
import sys

from tqdm import tqdm
from multi_taxi import single_taxi_v0, maps

from utils import ReplayBuffer, eval_agent, save_model, checkpoint_dir, map_preprocess
from agents import DQN, BfsAgent

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("episode_count", 30000, "Number of episodes to train for.")
flags.DEFINE_integer("train_episodes", 301, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 256, "Size of the training batch")
flags.DEFINE_float("target_period", 128, "How often to update the target net.")
flags.DEFINE_integer("replay_capacity", 2500, "Capacity of the replay buffer.")
flags.DEFINE_float("epsilon_begin", 1., "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 1000, "Steps over which to anneal eps.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_integer("evaluate_every", 1000,
                     "Number of episodes between evaluations.")


def run_episode(env, steps, agent, expert, replay_buffer, step_limit=100):
    obs_prev, _ = env.reset()
    dict_prev_obs = map_preprocess(env, obs_prev)
    episodic_reward = 0

    for _ in range(step_limit):
        action = expert(obs_prev)
        obs, reward, done, truncated, _ = env.step(action)
        dict_obs = map_preprocess(env, obs)
        steps += 1
        episodic_reward = reward + episodic_reward*FLAGS.discount_factor

        replay_buffer.push((dict_prev_obs, action, episodic_reward, FLAGS.discount_factor, dict_obs))
        obs_prev = obs

        if replay_buffer.is_ready() and steps % FLAGS.target_period == 0:
            batch = replay_buffer.sample()
            agent.learner_step(*batch)

        if done or truncated:
            break

    return steps

def train_loop(episode_count):
    env = single_taxi_v0.gym_env(
        num_passengers=3,
        has_standby_action=True,
        pickup_only=True,
        observation_type='symbolic',
        domain_map=maps.DEFAULT_MAP,
        render_mode='human'
    )
    env.seed(FLAGS.seed)
    env.reset()

    epsilon_cfg = dict(init_value=FLAGS.epsilon_begin, end_value=FLAGS.epsilon_end,
        transition_steps=FLAGS.epsilon_steps, power=1.)
    agent = DQN(env, epsilon_cfg, FLAGS.learning_rate)
    expert = BfsAgent(env)

    replay_buffer = ReplayBuffer(FLAGS.replay_capacity, FLAGS.batch_size)

    progress_bar = tqdm(range(episode_count))
    steps = 0

    for episode in range(episode_count):
        if episode % FLAGS.evaluate_every == 0:
            eval_rewards = eval_agent(env, agent, num_episodes=50)
            avg_reward = sum(eval_rewards)/len(eval_rewards)
            progress_bar.set_description(f'Avg Reward: {avg_reward:.2f}')

            save_model(agent.model, checkpoint_dir, '2_passenger_dqn')

        steps = run_episode(env, steps, agent, expert, replay_buffer)
        progress_bar.update(1)

if __name__ == "__main__":
    FLAGS(sys.argv)
    train_loop(FLAGS.episode_count)
