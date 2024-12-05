from multi_taxi import single_taxi_v0, maps
from tqdm import tqdm

from agents import BfsAgent, BCAgent
from utils import eval_agent, map_preprocess, ReplayBuffer, save_model, checkpoint_dir

def do_episode(env, step, agent, expert, replay_buffer, use_Dagger, max_steps=150, target_period=128):
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

    for _ in range(max_steps):
        expert_action = expert(obs)
        agent_action = agent(obs)
        action = agent_action if use_Dagger else expert_action

        learning_obs = map_preprocess(env, obs)
        replay_buffer.push((learning_obs, expert_action))
        obs, reward, done, truncated, _ = env.step(action)

        if step % target_period == 0 and replay_buffer.is_ready():
            batch_obs, expert_actions = replay_buffer.sample()
            agent.learner_step(batch_obs, expert_actions)

        step += 1
        total_reward += reward
        if done or truncated:
            break

    return step, total_reward

def main(num_episodes, model_name):
    env = single_taxi_v0.gym_env(
        num_passengers=2,
        max_fuel=75,
        max_steps=150,
        pickup_only=True,
        observation_type='symbolic',
        domain_map=maps.DEFAULT_MAP,
        render_mode='human')
    env.seed(42)
    env.reset()

    replay_buffer = ReplayBuffer(capacity=2500, batch_size=128)

    agent = BCAgent(env)
    expert = BfsAgent(env)

    progress_bar = tqdm(range(num_episodes))
    step_count = 0

    for episode in range(num_episodes):
        if episode % 1000 == 0:
            eval_rewards =  eval_agent(env, agent, num_episodes=50)
            avg_reward = sum(eval_rewards)/len(eval_rewards)
            progress_bar.set_description(f'Avg Reward: {avg_reward:.2f}')

            save_model(agent.model, checkpoint_dir, model_name)

        step_count, _ = do_episode(env, step_count, agent, expert, replay_buffer, use_Dagger=False)
        progress_bar.update(1)

    return agent

if __name__ == '__main__':
    main(30000, '2_passenger_bc')
