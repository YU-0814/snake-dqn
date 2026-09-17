import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BOARD = 8
GAMMA = 0.99


class CNN(nn.Module):
    """Two 3x3 convolutions on the (4, 8, 8) grid, then a flatten or global-average-pooling head."""

    def __init__(self, in_channels, n_actions, gap=False):
        super().__init__()
        self.gap = gap
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 if gap else 64 * BOARD * BOARD, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.mean([2, 3]) if self.gap else x.reshape(x.size(0), -1))


class Agent:
    """DQN with replay buffer, target network and linear epsilon decay."""

    def __init__(self, model, target, cfg, n_actions=4):
        self.n_actions = n_actions
        self.batch = cfg.get("batch", 64)
        self.train_start = cfg.get("train_start", 2000)
        self.eps, self.eps_min = 1.0, cfg.get("eps_min", 0.05)
        self.eps_dec = (self.eps - self.eps_min) / cfg.get("explore_step", 15000)
        self.target_update = cfg.get("target_update", 1000)
        self.steps = 0
        self.memory = deque(maxlen=cfg.get("memory", 50000))
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
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if self.eps > self.eps_min:
            self.eps -= self.eps_dec
        st, ac, rw, ns, dn = zip(*random.sample(self.memory, self.batch), strict=True)
        st, ns = torch.FloatTensor(np.stack(st)), torch.FloatTensor(np.stack(ns))
        ac, rw, dn = torch.LongTensor(ac), torch.FloatTensor(rw), torch.FloatTensor([int(d) for d in dn])
        pred = self.model(st).gather(1, ac.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            target = rw + (1 - dn) * GAMMA * self.target(ns).max(1)[0]
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target()
