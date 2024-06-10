import gymnasium as gym
import gym_wordle
from gym_wordle.envs.wordle_env import WordleEnv
from stable_baselines3 import A2C, PPO
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time
import matplotlib.pyplot as plt

env = Monitor(WordleEnv())

model = A2C.load("A2C_wordle")
# model = PPO.load("PPO_wordle")
models = {'A2C': A2C, 'PPO': PPO}

def simulate_game(model):
    obs, info = env.reset()
    done = False
    rewards = []
    num_iter = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)

        action = env.generate_valid_action()

        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        num_iter += 1
        env.render()

def run_games(model, num_games=100):
    env = WordleEnv()
    successes = 0

    for _ in range(num_games):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action = env.generate_valid_action()
            obs, reward, done, truncated, info = env.step(action)

        if reward > 0: 
            successes += 1

    success_ratio = successes / num_games
    return success_ratio

# for alg_name, alg in models.items():
#     num_games = 10000
#     success_ratio = run_games(model, num_games)

#     print(f"Success ratio over {num_games} games using {alg_name}: {success_ratio}")

simulate_game(model) # play a single game and display result