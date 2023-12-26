import gymnasium as gym
from models import DQNAgent


env = gym.make("highway-v0")
# value_iteration_agent = ValueIterationAgent(env)
# value_iteration_agent.train()

# q_learning_agent = QLearningAgent(env)
# q_learning_agent.train()
#
dqn_agent = DQNAgent(env)
dqn_agent.train()
#
# monte_carlo_agent = MonteCarloAgent(env)
# monte_carlo_agent.train()

print(1)