import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BOARD = 8
GAMMA = 0.99
RAY_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
FEATURE_DIM = 8 * 3 + 4


class CNN(nn.Module):
    """Two 3x3 convolutions on the (4, 8, 8) grid, then flatten or global average pooling."""

    def __init__(self, in_channels, n_actions, gap=False):
        super().__init__()
        self.gap = gap
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 if gap else 64 * BOARD * BOARD, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.mean([2, 3]) if self.gap else x.reshape(x.size(0), -1))


class MLP(nn.Module):
    """Two hidden layers on the 28-dim ray feature vector."""

    def __init__(self, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def ray_features(env, prev_action):
    """8 rays from the head: [food visible, 1/distance to body, 1/distance to wall], plus heading one-hot."""
    hr, hc = env.snake[-1]
    board, length, item = env.board, len(env.snake), env.ITEM
    out = []
    for dr, dc in RAY_DIRS:
        r, c, dist = hr + dr, hc + dc, 1
        see_food, dis_body = 0.0, 0.0
        while 0 <= r < BOARD and 0 <= c < BOARD:
            v = board[r, c]
            if v == item:
                see_food = 1.0
            elif 1 <= v <= length and dis_body == 0.0:
                dis_body = 1.0 / dist
            r, c, dist = r + dr, c + dc, dist + 1
        out += [see_food, dis_body, 1.0 / dist]
    heading = [0.0] * 4
    if 0 <= prev_action < 4:
        heading[prev_action] = 1.0
    return np.array(out + heading, dtype=np.float32)


def head_food_distance(env):
    hr, hc = env.snake[-1]
    food = np.argwhere(env.board == env.ITEM)
    return 0 if len(food) == 0 else abs(int(hr) - int(food[0][0])) + abs(int(hc) - int(food[0][1]))


class Agent:
    """DQN with replay, target network and linear epsilon decay; optional n-step returns and Double DQN."""

    def __init__(self, model, target, cfg, n_actions=4):
        self.n_actions = n_actions
        self.n_step = cfg.get("n_step", 1)
        self.double = cfg.get("double", False)
        self.batch = cfg.get("batch", 64)
        self.train_start = cfg.get("train_start", 2000)
        self.eps, self.eps_min = 1.0, cfg.get("eps_min", 0.05)
        self.eps_dec = (self.eps - self.eps_min) / cfg.get("explore_step", 20000)
        self.target_update = cfg.get("target_update", 1000)
        self.steps = 0
        self.memory = deque(maxlen=cfg.get("memory", 50000))
        self.nbuf = deque(maxlen=self.n_step)
        self.model, self.target = model, target
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.get("lr", 5e-4))
        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return int(self.model(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item())

    def remember(self, state, action, reward, next_state, done):
        self.nbuf.append((state, action, reward, next_state, done))
        if len(self.nbuf) >= self.n_step:
            self._flush()
        if done:
            while self.nbuf:
                self._flush()

    def _flush(self):
        """Pop the oldest transition, storing its n-step discounted return."""
        ret = 0.0
        for n, (_, _, r, _, done) in enumerate(self.nbuf, 1):
            ret += GAMMA ** (n - 1) * r
            if done:
                break
        next_state, done = self.nbuf[n - 1][3:]
        state, action = self.nbuf.popleft()[:2]
        self.memory.append((state, action, ret, next_state, done, n))

    def train(self):
        if self.eps > self.eps_min:
            self.eps -= self.eps_dec
        st, ac, rw, ns, dn, n = zip(*random.sample(self.memory, self.batch), strict=True)
        st, ns = torch.FloatTensor(np.stack(st)), torch.FloatTensor(np.stack(ns))
        ac, rw = torch.LongTensor(ac), torch.FloatTensor(rw)
        dn, n = torch.FloatTensor([int(d) for d in dn]), torch.FloatTensor(n)
        pred = self.model(st).gather(1, ac.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                nq = self.target(ns).gather(1, self.model(ns).argmax(1, keepdim=True)).squeeze(1)
            else:
                nq = self.target(ns).max(1)[0]
        loss = F.mse_loss(pred, rw + (1 - dn) * GAMMA ** n * nq)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target()
