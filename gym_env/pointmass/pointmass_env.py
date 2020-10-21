import gym
import numpy as np
from gym_env.pointmass.agent import Agent
import random


class PointMassEnv(gym.Env):
    def __init__(self, args, xlim=2, radius=1):
        self.xlim = xlim
        self.radius = radius
        self.dim = 2
        self.continuous_action = True
        self.max_acc = 2

        self.observation_shape = (4,)
        self.action_space = (5,) if not self.continuous_action else (2,)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space)
        self.constraint_maximum = 0
        self.dt = .05

        self.agent = Agent()
        self.agent.color = np.array([0.35, 0.35, 0.85])
        self.agent.position = None
        self.agent.orientation = None

        self.landmark = Agent()
        self.landmark.color = np.array([0, 0, 0])
        self.landmark.position = None
        self.landmark.orientation = None

    def _reset_agent(self):
        # self.agent.position = 0.1 * self.xlim * np.random.uniform(0, 1, self.dim) - self.xlim
        # self.landmark.position = self.xlim - 0.1 * self.xlim * np.random.uniform(0, 1, self.dim)

        self.agent.position = 0.1 * self.xlim * np.array([1,1]) - self.xlim
        self.landmark.position = - self.agent.position

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.counter = 0
        self._reset_agent()
        observations = self._get_obs()
        return observations

    def _get_obs(self):
        observations = np.concatenate((self.agent.position, self.landmark.position))
        return observations

    def step(self, action):
        pos = self.agent.position
        if not self.continuous_action:
            raise ValueError()
        else:
            action = np.clip(action, -self.max_acc, self.max_acc)
            if len(action.shape) == 2:
                action = action.flatten()
            pos += action * self.dt

        self.agent.position = pos
        next_obs = self._get_obs()

        reward = -np.linalg.norm(self.agent.position - self.landmark.position)

        obstacle_type = 1   # choose from 1~6
        if obstacle_type == 1: # one circle, binary cost
            cost = 0 if np.linalg.norm(self.agent.position) > self.radius and \
                np.max(np.abs(self.agent.position)) < self.xlim else 1
        elif obstacle_type == 2: # one smaller square, binary cost
            xlim_smaller = .5 * self.xlim
            cost = 0 if np.max(np.abs(self.agent.position)) > xlim_smaller and \
                        np.max(np.abs(self.agent.position)) < self.xlim else 1
        elif obstacle_type == 3: #two circles, binary cost
            center_1 = np.array([-.5, .5])
            center_2 = -center_1
            cost = 0 if np.linalg.norm(self.agent.position - center_1) > self.radius and \
                        np.linalg.norm(self.agent.position - center_2) > self.radius and \
                        np.max(np.abs(self.agent.position)) < self.xlim else 1
        elif obstacle_type == 4:  # one circle, continuous cost
            dist_to_center = np.linalg.norm(self.agent.position)
            cost = 0 if dist_to_center > self.radius and \
                        np.max(np.abs(self.agent.position)) < self.xlim else 2 * np.exp(-dist_to_center) + .5
        elif obstacle_type == 5:  # one smaller square, continuous cost
            dist_to_center = np.linalg.norm(self.agent.position)
            xlim_smaller = .5 * self.xlim
            cost = 0 if np.max(np.abs(self.agent.position)) > xlim_smaller and \
                        np.max(np.abs(self.agent.position)) < self.xlim else 2 * np.exp(-dist_to_center) + .5
        elif obstacle_type == 6:  # two circles, continuous cost
            dist_to_center = np.linalg.norm(self.agent.position)
            center_1 = np.array([-.5, .5])
            center_2 = -center_1
            cost = 0 if np.linalg.norm(self.agent.position - center_1) > self.radius and \
                        np.linalg.norm(self.agent.position - center_2) > self.radius and \
                        np.max(np.abs(self.agent.position)) < self.xlim else 2 * np.exp(-dist_to_center) + .5
        else:
            raise ValueError
        info = {}
        info['cost'] = cost

        done = True if np.linalg.norm(self.agent.position - self.landmark.position) <= 1e-1 else False

        self.counter += 1
        if self.counter >= 100:
            done = True

        return next_obs, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError
