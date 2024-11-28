from utils import eval_agent, MapWrapper
from multi_taxi import single_taxi_v0, maps

from agents import BfsAgent, BCAgent

def do_episode(env, agent, expert, max_steps=150):
    total_reward = 0
    obs, _ = env.reset()

    for _ in range(max_steps):
        action = agent(obs)
        expert_action = expert(obs)

        agent.learner_step(obs, expert_action)
        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward
        if done or truncated:
            break

    print('reward ', total_reward)

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

    agent = BCAgent(env, learning_rate=0.001)
    expert = BfsAgent(env)

    for episode in range(num_episodes):
        do_episode(env, agent, expert)

        if episode % 1000 == 0:
            print('reward ', eval_agent(env, agent))

if __name__ == '__main__':
    main()
