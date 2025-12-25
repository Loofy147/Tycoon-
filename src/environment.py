import numpy as np
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HardenedLegionEnv:
    def __init__(self, num_agents=20, grid_size=40, randomize=True):
        self.num_agents = num_agents
        self.bounds = 40.0
        self.grid_size = grid_size
        self.center = np.array([20.0, 20.0])
        self.comm_range = 15.0
        self.randomize = randomize

        self.friction = 0.98
        self.gravity_str = 0.1
        self.base_radius = 12.0
        self.base_angle = 0.0

        # Classes: 25% Scouts (0), 75% Miners (1)
        n_scouts = int(num_agents * 0.25)
        self.classes = np.array([0]*n_scouts + [1]*(num_agents-n_scouts))

        self.reset()

    def _calculate_base_pos(self):
        x = self.center[0] + self.base_radius * math.cos(self.base_angle)
        y = self.center[1] + self.base_radius * math.sin(self.base_angle)
        return np.array([x, y])

    def reset(self):
        self.positions = np.random.rand(self.num_agents, 2) * self.bounds
        self.velocities = np.zeros((self.num_agents, 2))
        self.energy = np.where(self.classes == 0, 2.0, 1.5)
        self.carrying = np.zeros(self.num_agents, dtype=bool)

        if self.randomize:
            self.walls = [
                {'x': np.random.uniform(5, 30), 'y': np.random.uniform(5, 30),
                 'w': np.random.uniform(2, 10), 'h': np.random.uniform(2, 10)}
                for _ in range(3)
            ]
        else:
            self.walls = [{'x': 15, 'y': 25, 'w': 10, 'h': 2}, {'x': 25, 'y': 10, 'w': 2, 'h': 10}]

        self.hazards = [{'pos': np.random.rand(2)*self.bounds, 'vel': (np.random.rand(2)-0.5)} for _ in range(10)]
        self.asteroids = [{'pos': np.random.rand(2)*self.bounds} for _ in range(40)]
        self.base_station = self._calculate_base_pos()

        return self.get_state_package()

    def get_adjacency_matrix(self):
        pos = torch.tensor(self.positions, device=device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dists = torch.norm(diff, dim=2)
        adj = (dists < self.comm_range).float()
        return adj

    def get_state_package(self):
        scale = self.grid_size / self.bounds
        global_grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 1: Danger
        for w in self.walls:
            x1, x2 = int(w['x']*scale), int((w['x']+w['w'])*scale)
            y1, y2 = int(w['y']*scale), int((w['y']+w['h'])*scale)
            x1, x2 = max(0, x1), min(self.grid_size, x2)
            y1, y2 = max(0, y1), min(self.grid_size, y2)
            global_grid[1, y1:y2, x1:x2] = 1.0
        for h in self.hazards:
            px, py = int(h['pos'][0]*scale), int(h['pos'][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: global_grid[1, py, px] = 1.0

        # Channel 2: Reward
        bx, by = int(self.base_station[0]*scale), int(self.base_station[1]*scale)
        if 0<=bx<self.grid_size and 0<=by<self.grid_size: global_grid[2, by, bx] = 1.0
        for a in self.asteroids:
            px, py = int(a['pos'][0]*scale), int(a['pos'][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: global_grid[2, py, px] = 0.5

        observations = []
        for i in range(self.num_agents):
            local = global_grid.copy()
            val = 1.0 if self.classes[i] == 0 else 0.5
            px, py = int(self.positions[i][0]*scale), int(self.positions[i][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: local[0, py, px] = val
            observations.append(local)

        return torch.FloatTensor(np.array(observations)).to(device), self.get_adjacency_matrix(), torch.LongTensor(self.classes).to(device)

    def step(self, actions):
        self.base_angle += 0.02
        self.base_station = self._calculate_base_pos()

        thrust_mult = np.where(self.classes == 0, 1.5, 0.8)
        self.velocities += actions * thrust_mult[:, np.newaxis]

        for i in range(self.num_agents):
            to_center = self.center - self.positions[i]
            dist = np.linalg.norm(to_center)
            if dist > 1.0:
                self.velocities[i] += to_center / dist * (self.gravity_str / (dist/5.0))

        self.velocities *= self.friction
        new_pos = self.positions + self.velocities

        for i in range(self.num_agents):
            px, py = new_pos[i]
            hit = False
            for w in self.walls:
                if w['x'] < px < w['x']+w['w'] and w['y'] < py < w['y']+w['h']: hit = True
            if hit: self.velocities[i] *= -0.5
            else: self.positions[i] = new_pos[i]

        self.positions = np.clip(self.positions, 0, self.bounds)

        for h in self.hazards:
            h['pos'] += h['vel']
            for k in range(2):
                if h['pos'][k] < 0 or h['pos'][k] > self.bounds: h['vel'][k] *= -1

        thrust_mag = np.linalg.norm(actions, axis=1)
        self.energy -= (0.005 * thrust_mag) + 0.0005

        rewards = np.zeros(self.num_agents)
        minerals = 0

        # --- THE ECONOMIC FIX ---
        for a in self.asteroids:
            d = np.linalg.norm(self.positions - a['pos'], axis=1)
            mask = (d < 1.5) & (~self.carrying) & (self.classes == 1)
            if np.any(mask):
                winner = np.where(mask)[0][0]
                self.carrying[winner] = True
                a['pos'] = np.random.rand(2)*self.bounds
                rewards[winner] += 2.0 # INFLATION: 2.0

        d_base = np.linalg.norm(self.positions - self.base_station, axis=1)
        mask_dep = (d_base < 2.0) & (self.carrying)
        if np.any(mask_dep):
            self.carrying[mask_dep] = False
            self.energy[mask_dep] = 1.0
            rewards[mask_dep] += 10.0 # JACKPOT: 10.0
            minerals += np.sum(mask_dep)

        # Add a base reward for miners who return to the base
        mask_base = (d_base < 2.0) & (self.classes == 1)
        rewards[mask_base] += 0.1

        for h in self.hazards:
            d = np.linalg.norm(self.positions - h['pos'], axis=1)
            mask_hit = d < 1.0
            if np.any(mask_hit):
                rewards[mask_hit] -= 5.0
                self.energy[mask_hit] -= 0.5

        dead = self.energy <= 0
        if np.any(dead):
            self.positions[dead] = self.base_station.copy()
            self.velocities[dead] = 0
            self.energy[dead] = 1.0
            self.carrying[dead] = False
            rewards[dead] -= 5.0 # Reduced from -20

        return self.get_state_package(), rewards, minerals