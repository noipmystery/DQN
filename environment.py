import gym
from gym.wrappers import AtariPreprocessing, FrameStack


def make_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True)
    env = FrameStack(env, num_stack=4)
    return env
