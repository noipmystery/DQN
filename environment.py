import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation


def make_env(env_name):
    env = gym.make(env_name)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    return env
