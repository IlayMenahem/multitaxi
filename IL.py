from multi_taxi import single_taxi_v0, maps
from tqdm import tqdm

from agents import BfsAgent, BCAgent
from utils import eval_agent, MapWrapper, ReplayBuffer

def do_episode(env, agent, expert, replay_buffer, use_Dagger, max_steps=150, target_period=32):
    '''
    does imitation learning episode
    :param env: environment
    :param agent: agent to train
    :param expert: expert agent
    :param replay_buffer: replay buffer
    :param use_Dagger: whether to use Dagger or behavior cloning
    :param max_steps: maximum number of steps in episode
    :param target_period: how often to update agent
    '''
    total_reward = 0
    obs, _ = env.reset()

    for step in range(max_steps):
        expert_action = expert(obs)
        agent_action = agent(obs)
        action = agent_action if use_Dagger else expert_action

        replay_buffer.push((obs, expert_action))
        obs, reward, done, truncated, _ = env.step(action)

        if step % target_period == 0 and replay_buffer.is_ready():
            batch_obs, expert_actions = replay_buffer.sample()
            agent.learner_step(batch_obs, expert_actions)

        total_reward += reward
        if done or truncated:
            break

def main(num_episodes=15000):
    env = single_taxi_v0.gym_env(
        num_passengers=3,
        max_fuel=75,
        max_steps=150,
        pickup_only=True,
        observation_type='symbolic',
        domain_map=maps.DEFAULT_MAP,
        render_mode='human'
    )
    env.seed(42)
    env = MapWrapper(env)

    replay_buffer = ReplayBuffer(capacity=1000, batch_size=32)

    agent = BCAgent(env, learning_rate=0.001)
    expert = BfsAgent(env)

    progress_bar = tqdm(range(num_episodes))

    for episode in range(num_episodes):
        if episode % 100 == 0:
            eval_rewards =  eval_agent(env, agent)
            avg_reward = sum(eval_rewards)/len(eval_rewards)

            progress_bar.set_description(f'Avg Reward: {avg_reward:.2f}')
            progress_bar.update(100)

        do_episode(env, agent, expert, replay_buffer, use_Dagger=False)

if __name__ == '__main__':
    main()
