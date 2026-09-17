"""python train.py configs/<name>.json  ->  results/<name>/{cfg.json, history.npz, model.pth}"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from dqn import BOARD, GAMMA, CNN, MLP, Agent, head_food_distance, ray_features
from snake_env import SnakeGameEnv

INIT_LENGTH = 3
MAX_STEPS_WITHOUT_FOOD = BOARD ** 2


def kaiming(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")


def run(cfg):
    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("threads", 8))
    use_cnn = cfg.get("encoder", "cnn") == "cnn"
    env = SnakeGameEnv(board_size=BOARD, n_channel=4 if use_cnn else 1, n_target=1,
                       death_penalty=cfg.get("death", -10.0), step_reward=cfg.get("step", -0.01),
                       target_reward=cfg.get("target", 1.0))
    if use_cnn:
        model, target = CNN(4, 4, cfg.get("gap", False)), CNN(4, 4, cfg.get("gap", False))
        model.apply(kaiming)
    else:
        model, target = MLP(4), MLP(4)
    agent = Agent(model, target, cfg)
    pbrs = cfg.get("pbrs", False)

    def state_of(obs, info, action):
        return (obs > 0).astype(np.float32) if use_cnn else ray_features(env, info.get("prev_action", action))

    out = os.path.join("results", cfg["name"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    scores, t0 = [], time.time()
    for ep in range(1, cfg.get("episodes", 3000) + 1):
        obs, info = env.reset(seed=seed * 100000 + ep)
        state = state_of(obs, info, -1)
        agent.nbuf.clear()
        done, no_food_steps, length = False, 0, info["snake_length"]
        phi = -head_food_distance(env) if pbrs else 0.0
        while not done:
            action = agent.act(state)
            obs, reward, terminated, _, info = env.step(action)
            next_state = state_of(obs, info, action)
            ate = info["snake_length"] > length
            if pbrs:
                phi_next = -head_food_distance(env)
                if not ate:
                    reward += GAMMA * phi_next - phi
                phi = phi_next
            no_food_steps = 0 if ate else no_food_steps + 1
            length = info["snake_length"]
            done = terminated or no_food_steps >= MAX_STEPS_WITHOUT_FOOD
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.train_start:
                agent.train()
            state = next_state
        scores.append(length - INIT_LENGTH)
        if ep % 100 == 0:
            print(f"[{cfg['name']}] ep {ep} avg100 {np.mean(scores[-100:]):.2f} eps {agent.eps:.3f}", flush=True)

    torch.save(agent.model.state_dict(), os.path.join(out, "model.pth"))
    np.savez(os.path.join(out, "history.npz"), scores=scores)
    print(f"[{cfg['name']}] avg300 {np.mean(scores[-300:]):.2f} max {max(scores)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        run(json.load(f))
