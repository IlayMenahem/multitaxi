from multi_taxi import single_taxi_v0, maps

from agents import BfsAgent, BCAgent
from utils import eval_agent, MapWrapper, ReplayBuffer

def do_episode(env, agent, expert, replay_buffer, max_steps=150, target_period=32):
    total_reward = 0
    obs, _ = env.reset()

    for step in range(max_steps):
        expert_action = expert(obs)
        replay_buffer.push((obs, expert_action))
        obs, reward, done, truncated, _ = env.step(expert_action)

        if step % target_period == 0 and replay_buffer.is_ready():
            batch_obs, expert_actions = replay_buffer.sample()
            agent.learner_step(batch_obs, expert_actions)

        total_reward += reward
        if done or truncated:
            break

def main(num_episodes=1000000):
    env = single_taxi_v0.gym_env(
        num_passengers=3,
        max_fuel=75,
        max_steps=150,
        pickup_only=True,
        observation_type='symbolic',
        domain_map=maps.DEFAULT_MAP,
        render_mode='human'
    )
    env = MapWrapper(env)

    replay_buffer = ReplayBuffer(capacity=1000, batch_size=32)

    agent = BCAgent(env, learning_rate=0.001)
    expert = BfsAgent(env)

    for episode in range(num_episodes):
        do_episode(env, agent, expert, replay_buffer)

        if episode % 100 == 0:
            print('eval rewards ', eval_agent(env, agent))

if __name__ == '__main__':
    main()
