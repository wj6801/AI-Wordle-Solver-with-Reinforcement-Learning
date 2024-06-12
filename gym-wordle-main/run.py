import gymnasium as gym
import gym_wordle
from gym_wordle.envs.wordle_env import WordleEnv
from stable_baselines3 import A2C, PPO
import numpy as np
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

simulate_game(model) # play a single game and display result