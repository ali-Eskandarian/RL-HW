import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)

    def train(self):
        delta = np.inf
        while delta > self.theta:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                self.V[s] = np.max([sum([p*(r + self.gamma*self.V[s_]) for p, s_, r, _ in self.env.P[s][a]]) for a in range(self.env.action_space.n)])
                delta = max(delta, abs(v - self.V[s]))

    def act(self, state):
        return np.argmax([sum([p*(r + self.gamma*self.V[s_]) for p, s_, r, _ in self.env.P[state][a]]) for a in range(self.env.action_space.n)])

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.max(self.Q[next_state]) * (1 - done)
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * target

    def train(self, episodes=1000, max_steps=500):
        for episode in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01, alpha=0.001, batch_size=4, memory_size=10000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        for i in states:
            if len(i) != 5:
                print(i)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.model(next_states).detach().max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, episodes=1000, max_steps=500):
        for episode in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                action = self.act(state)
                x = self.env.step(action)
                next_state, reward, terminated, truncated , info, = self.env.step(action)
                done = terminated or truncated
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
                self.learn()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MonteCarloAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}
        self.policy = np.zeros(env.observation_space.n, dtype=int)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.policy[state]

    def learn(self, episode):
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                self.policy[state] = np.argmax(self.Q[state])

    def train(self, episodes=1000, max_steps=500):
        for episode in range(episodes):
            state = self.env.reset()
            episode = []
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break
            self.learn(episode)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
