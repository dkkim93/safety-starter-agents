import gym
import numpy as np


class PendulumCostWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(PendulumCostWrapper, self).__init__(env)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = 1.  # Default value from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        l = 1.  # Default value from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        dt = self.dt
        self.max_torque = 15.

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111
        self.state = np.array([newth, newthdot])

        bound = 1
        newth_ = angle_normalize(newth)
        constraint = .1 if newth_ > bound or newth_ < -bound else 0
        info = {'cost': constraint}

        self.counter += 1
        done = False
        if self.counter >= 100:
            done = True

        return self._get_obs(), -costs, done, info

    def reset(self):
        self.counter = 0
        self.state = 0.5 * np.array([np.pi, 1])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot])
        if len(obs.shape) == 2:
            obs = obs.flatten()
        return obs


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
