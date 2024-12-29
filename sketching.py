from multi_taxi import single_taxi_v0, maps
from agents import BCAgent
from utils import eval_agent

if __name__ == '__main__':
    env = single_taxi_v0.gym_env(
        num_passengers=2,
        pickup_only=True,
        observation_type='symbolic',
        domain_map=maps.DEFAULT_MAP,
        render_mode='human'
    )
    env.seed(0)
    env.reset()

    agent = BCAgent(env, checkpoint_name='2_passenger_bc')
    eval_agent(env, agent, num_episodes=1)
