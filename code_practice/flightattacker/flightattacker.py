import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from collections import deque

# === 自定义 FlightAttacker 环境 ===
class FlightAttackerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FlightAttackerEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # 上 下 左 右 发射
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)

    def reset(self):
        self.player_pos = [50, 10]
        self.enemy_pos = [random.randint(0, 100), 90]
        self.bullets = []
        self.timestep = 0
        return self._get_obs()

    def _get_obs(self):
        bullet_x, bullet_y = self.bullets[0] if self.bullets else (0, 0)
        return np.array([
            self.player_pos[0], self.player_pos[1],
            self.enemy_pos[0], self.enemy_pos[1],
            bullet_x, bullet_y
        ], dtype=np.float32)

    def step(self, action):
        self.timestep += 1
        if action == 0: self.player_pos[1] += 5
        elif action == 1: self.player_pos[1] -= 5
        elif action == 2: self.player_pos[0] -= 5
        elif action == 3: self.player_pos[0] += 5
        elif action == 4: self.bullets.append(self.player_pos.copy())

        self.player_pos[0] = np.clip(self.player_pos[0], 0, 100)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, 100)

        self.bullets = [[x, y + 10] for x, y in self.bullets if y <= 100]

        reward = 0
        done = False
        for bullet in self.bullets:
            if np.linalg.norm(np.array(bullet) - np.array(self.enemy_pos)) < 10:
                reward = 10
                done = True

        if self.timestep >= 200:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Player: {self.player_pos}, Enemy: {self.enemy_pos}, Bullets: {self.bullets}")

# === Q网络 ===
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# === DQN Agent ===
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.learn_step = 0
        self.action_dim = action_dim

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def remember(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# === 主训练程序 ===
if __name__ == "__main__":
    env = FlightAttackerEnv()
    agent = DQNAgent(state_dim=6, action_dim=5)

    EPISODES = 300
    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
