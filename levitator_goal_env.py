import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import gym

class Levitator(gym.GoalEnv):
    def __init__(self, config, initial_state, target, delay, dt=0.0001, action_type="SVD", sample_target=False):
        # ----- Transducer locations -----
        self.config = config
        self.target = target

        if (self.config == 'two'):
            # One transducer above and below
            self.locations = np.array([[0., 0., 0.05], [0., 0., -0.05]])
        elif (self.config == 'six'):
            # Two transducers in x,y and z directions
            self.locations = np.array([[0., 0., 0.05], [0., 0., -0.05],
                                       [0.05, 0., 0.], [-0.05, 0., 0.],
                                       [0., 0.05, 0.], [0., -0.05, 0.]])
        elif (self.config == 'eight'):
            # Four transducers above and below
            self.locations = np.array(
                [[0.01, 0.01, 0.05], [0.01, -0.01, 0.05], [-0.01, 0.01, 0.05], [-0.01, -0.01, 0.05],
                 [0.01, 0.01, -0.05], [0.01, -0.01, -0.05], [-0.01, 0.01, -0.05], [-0.01, -0.01, -0.05]])
        elif (self.config == 'ten'):
            # Four transducers above and below
            self.locations = np.array(
                [[0.02, 0.02, 0.05], [0.00, 0.00, 0.05], [0.02, -0.02, 0.05], [-0.02, 0.02, 0.05], [-0.02, -0.02, 0.05],
                 [0.02, 0.02, -0.05],[0.00, 0.00, -0.05], [0.02, -0.02, -0.05], [-0.02, 0.02, -0.05], [-0.02, -0.02, -0.05]])

        elif (self.config == 'pyramid'):
            # Four transducers in a pyramid shape
            self.locations = np.array(
                [[0.0471, 0., -0.0166], [-0.0236, 0.0408, -0.0166], [-0.0236, -0.0408, -0.0166], [0., 0., 0.05]])
        elif (self.config == 'triangles' or self.config == 'full'):
            # Use triangles consisting of 15 transducers each or use the full configuration of levitator
            self.locations_tx = np.loadtxt('tx_locations.txt', delimiter=',') / 100  # All transducers, cm -> m
            self.locations = np.array(self.locations_tx, ndmin=2)

        self.locations_tensor = torch.from_numpy(self.locations)

        self.n_transducers = self.locations.shape[0]

        # ----- Constants -----
        # Radius of drop
        self.r = 0.0005  # m
        # Density of water
        self.rho_water = 30  # kg/m^3
        # Density of air
        self.rho = 1.225  # kg/m^3
        # Volume of drop
        self.V = 4 / 3 * np.pi * self.r ** 3
        # Mass of drop
        self.m = self.V * self.rho_water
        # Speed of sound in water
        self.c_water = 1480  # m/s
        # Speed of sound in air
        self.c = 343  # m/s

        # Gravitational acceleration
        self.g = 9.81  # m/s^2

        # Timestep of 0.1 ms
        self.dt = dt  # s

        # Constants of transducers
        self.frequency = 40e3
        self.omega = 2*np.pi*self.frequency
        self.p0 = 40  # Pascals
        self.wavelength = self.c / self.frequency

        # Amplitude
        # amp_array = np.array([2.93348267, 0.93348267, 1., 1.])
        # self.init_amps = amp_array / np.linalg.norm(amp_array)
        # self.amplitude = torch.from_numpy(self.init_amps)
        self.amplitude = torch.ones(self.n_transducers)

        # Force
        self.F = torch.zeros(3)
        self.trap = torch.zeros(1)

        # One of: "raw" means full transducer space, 
        # "SVD" means coordinate and angles of next 
        self.action_type = action_type

        if action_type == "SVD":
            # The agent gives actions from -1 to 1, and they are scaled to the proper intervals below
            high = torch.ones(6)
            low = -torch.ones(6)

            self.action_space = gym.spaces.box.Box(low=low.numpy(), high=high.numpy())
        else:
            raise NotImplementedError()

        self.initial_state = initial_state.clone()
        self.state = initial_state.clone()
        
        low = torch.concat([initial_state[:3] - (initial_state[:3] * 0.5), -torch.ones(3)]).numpy()
        high = torch.concat([target + (target * 0.5), torch.ones(3)]).numpy()

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=low, high=high),
            "desired_goal": gym.spaces.Box(low=low[:3], high=high[:3]),
            "achieved_goal": gym.spaces.Box(low=low[:3], high=high[:3])
        })

        self.sample_target = sample_target
        self.delay = delay

    def reset(self):
        self.state = self.initial_state.clone() 

        if self.sample_target:
            #self.state += abs(0.005 * torch.randn_like(self.initial_state)) # sample initial too
            self.target = self.initial_state.clone()[:3] + abs(0.005 * torch.randn_like(self.target))

        return {
            "observation": self.state,
            "desired_goal": self.target,
            "achieved_goal": self.state[:3]
        }

    def step(self, action):
        action = self._rescale_actions(action)
        action = np.concatenate([action, np.ones(1), 0.3 * np.ones(1)], axis=0)
        action = self._SVD(action.reshape(1, -1))

        state = self.state.reshape(1, -1).clone()

        for t in range(self.delay):
            state = self._dynamics(state, action)

        reward = self.reward(state.squeeze())
        self.state = state.squeeze().clone()

        terminated = False

        if torch.linalg.norm(self.state[:3] - self.target) > 0.1:
            reward = -1000
            terminated = True

        obs = {
            "observation": self.state,
            "desired_goal": self.target,
            "achieved_goal": self.state[:3]
        }

        reward = self.compute_reward(self.state, self.target, {})

        return obs, reward, terminated, {}

    def _rescale_actions(self, action):
        # assume action is -1 to 1
        return np.concatenate([action[:3] * 0.01, (action[3:] * 180) + 90])

    def compute_reward(self, achieved_goal: object, desired_goal: object, info) -> float:
        if torch.is_tensor(achieved_goal):
            achieved_goal, desired_goal = achieved_goal.numpy(), desired_goal.numpy()
        
        distance = np.linalg.norm(achieved_goal[..., :3] - desired_goal[..., :3], axis=-1)
        asd = -(distance > 0.01).astype(np.float32)
        return asd
        
    def _dynamics(self, state, action):
        # Force from transducers
        F = self._get_force_vec(state, action)

        # Substract gravity in the z-direction
        F[:, 2] -= self.m * self.g

        # Update velocities and location
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        v = state[:, 3:] + F / self.m * self.dt
        coord = state[:, 0:3] + v * self.dt

        return torch.cat([coord, v], dim=1)

    def _SVD(self, X_act=np.array([[0, 0, 0, 0, 0, 0, 1, 0.5]])):

        N_tx = self.locations.shape[0]
        N_act = X_act.shape[0]
        M = np.zeros((N_act, 4, N_tx), dtype=np.complex128)
        b = np.zeros((N_act, 4), dtype=np.complex128)

        rot = Rotation.from_euler('xyz', X_act[:, 3:6].repeat(3, axis=0), degrees=True)
        orientation = rot.apply((np.eye(3) * np.ones((N_act, 1, 1))).reshape((-1, 3))).reshape(-1, 3, 3)

        # set pressure node
        b[: 0] = 0
        # match pressure gradient along z and x
        b[:, 1:4] = X_act[:, 6, np.newaxis] * (orientation[:, 2] + 1j * X_act[:, 7, np.newaxis] * orientation[:, 1])

        waveDir = X_act[:, np.newaxis, :3] - self.locations

        distance = np.linalg.norm(self.locations - X_act[:, np.newaxis, :3], axis=2) - np.linalg.norm(self.locations, axis=1)
        pressure_tx = np.exp(1j * 2 * np.pi * distance / self.wavelength)

        M[:, 0] = pressure_tx
        M[:, 1:4, :] = (1j * np.transpose(waveDir, axes=(0, 2, 1)) / np.linalg.norm(waveDir, axis=2)[:, np.newaxis,
                                                                     :] * pressure_tx[:, np.newaxis,
                                                                          :])  # .reshape(N_act, 3, N_tx)
        M_inv = np.linalg.pinv(M)

        amps = (M_inv * b[:, np.newaxis, :]).sum(axis=2)

        return torch.tensor(amps / np.abs(amps).max(axis=1)[:, np.newaxis])

    def _get_force_vec(self, state, action):
        if state.ndim == 1:
            point = state[0:3].view(1, -1).expand(action.shape[0], -1)
        else:
            point = state[:, 0:3]
        point.requires_grad_(True)

        phase = torch.angle(action)
        amplitude = torch.abs(action)

        distances = torch.linalg.norm(self.locations_tensor.unsqueeze(2) - point.T, dim=1).T \
                    - torch.linalg.norm(self.locations_tensor, dim=1)

        pressure_re = (torch.cos(2 * torch.pi * distances / self.wavelength + phase) * amplitude * self.p0).sum(dim=1)
        pressure_im = (torch.sin(2 * torch.pi * distances / self.wavelength + phase) * amplitude * self.p0).sum(dim=1)

        pressure_re_dx = torch.autograd.grad(pressure_re, point, grad_outputs=torch.ones_like(pressure_re), create_graph=True)[0]
        pressure_im_dx = torch.autograd.grad(pressure_im, point, grad_outputs=torch.ones_like(pressure_re), create_graph=True)[0]

        # Gor'kov potential
        inner = 1 / (2 * self.rho * self.c ** 2) * (pressure_re.square() + pressure_im.square())
        other = 3 / (4 * self.rho * self.omega ** 2) * (pressure_re_dx.square() + pressure_im_dx.square()).sum(dim=1)
        U = self.V * (inner - other)

        grad_U = torch.autograd.grad(U, point, grad_outputs=torch.ones(action.shape[0]), create_graph=True)[0]

        # Force
        F = - grad_U.detach()

        return F

# Basic usage
if __name__ == '__main__':
    initial = torch.tensor([-0.02, 0.0, -0.01, 0, 0, 0])
    target = torch.tensor([0.02, 0.0, 0.01])

    nb_steps = 10000
    delay = 10
    config = "full"
    planner_dynamics = "full"

    env = Levitator(config, initial, target, delay)

    for e in range(3):
        observation, info = env.reset()

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            print(reward)

            if terminated or truncated:
                observation, info = env.reset()