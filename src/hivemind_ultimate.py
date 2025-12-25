import numpy as np
import random
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import sys

# ==========================================
# 0. CONFIGURATION & UTILS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing HIVEMIND ULTIMATE (Memory + Curriculum) on: {device}")

if not os.path.exists('outputs'):
    os.makedirs('outputs')

class RunningMeanStd:
    """Dynamically normalizes rewards to Mean=0, Std=1.
    This removes the need for manual reward tuning."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.merge_stats(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def merge_stats(self, mean_a, var_a, count_a, mean_b, var_b, count_b):
        delta = mean_b - mean_a
        m_a = var_a * count_a
        m_b = var_b * count_b
        M2 = m_a + m_b + np.square(delta) * count_a * count_b / (count_a + count_b)
        new_count = count_a + count_b
        new_mean = mean_a + delta * count_b / new_count
        new_var = M2 / new_count
        return new_mean, new_var, new_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

# ==========================================
# 1. ENVIRONMENT: THE CURRICULUM
# ==========================================
class UltimateLegionEnv:
    def __init__(self, num_agents=20, grid_size=40, difficulty=0.0):
        self.num_agents = num_agents
        self.bounds = 40.0
        self.grid_size = grid_size
        self.center = np.array([20.0, 20.0])
        self.comm_range = 15.0

        # Difficulty Parameters (0.0 to 1.0)
        self.difficulty = difficulty
        self.set_difficulty(difficulty)

        # Classes: 0=Scout, 1=Miner
        n_scouts = int(num_agents * 0.3) # More scouts for complex maps
        self.classes = np.array([0]*n_scouts + [1]*(num_agents-n_scouts))

        self.reset()

    def set_difficulty(self, d):
        self.difficulty = np.clip(d, 0.0, 1.0)
        # Scale Physics based on Difficulty
        self.gravity_str = 0.0 + (0.15 * self.difficulty) # 0.0 -> 0.15
        self.num_hazards = 5 + int(10 * self.difficulty)  # 5 -> 15
        self.hazard_speed = 0.2 + (0.8 * self.difficulty) # 0.2 -> 1.0
        self.base_speed = 0.0 + (0.04 * self.difficulty)  # Static -> Fast Orbit

    def _calculate_base_pos(self):
        x = self.center[0] + self.base_radius * math.cos(self.base_angle)
        y = self.center[1] + self.base_radius * math.sin(self.base_angle)
        return np.array([x, y])

    def reset(self):
        self.positions = np.random.rand(self.num_agents, 2) * self.bounds
        self.velocities = np.zeros((self.num_agents, 2))
        self.energy = np.where(self.classes == 0, 2.5, 2.0) # Boosted reserves
        self.carrying = np.zeros(self.num_agents, dtype=bool)

        self.base_radius = 12.0
        self.base_angle = 0.0
        self.base_station = self._calculate_base_pos()

        # Dynamic Map Generation
        self.walls = [
            {'x': 10, 'y': 20, 'w': 20, 'h': 2}, # Central barrier
            {'x': 20, 'y': 5, 'w': 2, 'h': 10}
        ]

        self.hazards = [{'pos': np.random.rand(2)*self.bounds, 'vel': (np.random.rand(2)-0.5)*self.hazard_speed} for _ in range(self.num_hazards)]
        self.asteroids = [{'pos': np.random.rand(2)*self.bounds} for _ in range(40)]

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

        # 1. Danger Channel
        for w in self.walls:
            x1, x2 = int(w['x']*scale), int((w['x']+w['w'])*scale)
            y1, y2 = int(w['y']*scale), int((w['y']+w['h'])*scale)
            x1, x2 = max(0, x1), min(self.grid_size, x2)
            y1, y2 = max(0, y1), min(self.grid_size, y2)
            global_grid[1, y1:y2, x1:x2] = 1.0
        for h in self.hazards:
            px, py = int(h['pos'][0]*scale), int(h['pos'][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: global_grid[1, py, px] = 1.0

        # 2. Reward Channel
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
        self.base_angle += self.base_speed
        self.base_station = self._calculate_base_pos()

        thrust_mult = np.where(self.classes == 0, 1.5, 0.8)
        self.velocities += actions * thrust_mult[:, np.newaxis]

        # Gravity (Dynamic based on difficulty)
        if self.difficulty > 0.1:
            for i in range(self.num_agents):
                to_center = self.center - self.positions[i]
                dist = np.linalg.norm(to_center)
                if dist > 1.0:
                    self.velocities[i] += to_center / dist * (self.gravity_str / (dist/5.0))

        self.velocities *= 0.98 # Friction
        new_pos = self.positions + self.velocities

        # Walls
        for i in range(self.num_agents):
            px, py = new_pos[i]
            hit = False
            for w in self.walls:
                if w['x'] < px < w['x']+w['w'] and w['y'] < py < w['y']+w['h']: hit = True
            if hit: self.velocities[i] *= -0.5
            else: self.positions[i] = new_pos[i]

        self.positions = np.clip(self.positions, 0, self.bounds)

        # Hazards
        for h in self.hazards:
            h['pos'] += h['vel']
            for k in range(2):
                if h['pos'][k] < 0 or h['pos'][k] > self.bounds: h['vel'][k] *= -1

        thrust_mag = np.linalg.norm(actions, axis=1)
        self.energy -= (0.005 * thrust_mag) + 0.0005

        rewards = np.zeros(self.num_agents)
        minerals = 0

        # Mining
        for a in self.asteroids:
            d = np.linalg.norm(self.positions - a['pos'], axis=1)
            mask = (d < 1.5) & (~self.carrying) & (self.classes == 1)
            if np.any(mask):
                winner = np.where(mask)[0][0]
                self.carrying[winner] = True
                a['pos'] = np.random.rand(2)*self.bounds
                rewards[winner] += 10.0 # Standardized Reward

        # Deposit
        d_base = np.linalg.norm(self.positions - self.base_station, axis=1)
        mask_dep = (d_base < 2.0) & (self.carrying)
        if np.any(mask_dep):
            self.carrying[mask_dep] = False
            self.energy[mask_dep] = 2.0 # Full Recharge + Boost
            rewards[mask_dep] += 50.0 # Big Reward
            minerals += np.sum(mask_dep)

        # Hazards
        for h in self.hazards:
            d = np.linalg.norm(self.positions - h['pos'], axis=1)
            mask_hit = d < 1.0
            if np.any(mask_hit):
                rewards[mask_hit] -= 5.0
                self.energy[mask_hit] -= 0.5

        # Death
        dead = self.energy <= 0
        if np.any(dead):
            self.positions[dead] = self.base_station.copy()
            self.velocities[dead] = 0
            self.energy[dead] = 1.0
            self.carrying[dead] = False
            rewards[dead] -= 20.0

        return self.get_state_package(), rewards, minerals

# ==========================================
# 2. ULTIMATE LEGION NET (Vision + Memory + Telepathy)
# ==========================================
class SwarmAttentionHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embeddings, adjacency):
        Q = self.W_q(embeddings)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)
        scores = torch.matmul(Q, K.t()) / math.sqrt(self.embed_dim)
        mask = (adjacency == 0)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        return self.out_proj(context)

class UltimateLegionNet(nn.Module):
    def __init__(self, c_in=3, n_actions=5, grid_size=40, hidden_dim=128):
        super().__init__()

        # 1. Vision (CNN)
        self.conv1 = nn.Conv2d(c_in, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)

        # Dynamic Size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(grid_size, 5, 2), 3, 2)
        convh = conv2d_size_out(conv2d_size_out(grid_size, 5, 2), 3, 2)
        flat_size = convw * convh * 32

        self.fc_vis = nn.Linear(flat_size, hidden_dim)
        self.class_embed = nn.Embedding(2, 16)

        # 2. Memory (GRU) - The Upgrade
        # Input: Vision (128) + Class (16) = 144
        self.gru = nn.GRUCell(hidden_dim + 16, hidden_dim)

        # 3. Telepathy (GNN)
        self.attention = SwarmAttentionHead(embed_dim=hidden_dim)

        # 4. Decision
        # Input: Memory (128) + Telepathy (128) = 256
        self.fc_q = nn.Linear(hidden_dim * 2, n_actions)

        self.register_buffer('rho', torch.tensor(0.0))
        self.hidden_dim = hidden_dim

    def forward(self, img_batch, adjacency, class_ids, hidden_state):
        # Vision
        x = F.relu(self.conv1(img_batch))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        vis_feat = F.relu(self.fc_vis(x))

        # Class
        cls_feat = self.class_embed(class_ids)

        # Combine for Memory
        features = torch.cat([vis_feat, cls_feat], dim=1)

        # Memory Step
        new_hidden = self.gru(features, hidden_state)

        # Telepathy (Share Memories, not just sights)
        swarm_context = self.attention(new_hidden, adjacency)

        # Decision
        combined = torch.cat([new_hidden, swarm_context], dim=1)
        q_vals = self.fc_q(combined)

        return q_vals, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(device)

# ==========================================
# 3. META-TRAINER (Curriculum Aware)
# ==========================================
class HiveMind:
    def __init__(self):
        self.population = []
        for _ in range(2):
            self.population.append({
                "params": { "lr": 0.0003, "rho_alpha": 0.01 },
                "score": 0, "model_state": None
            })
        self.reward_normalizer = RunningMeanStd(shape=())

    def get_model(self, params):
        policy = UltimateLegionNet(grid_size=40).to(device)
        target = UltimateLegionNet(grid_size=40).to(device)
        target.load_state_dict(policy.state_dict())
        optimizer = optim.Adam(policy.parameters(), lr=params['lr'])
        return policy, target, optimizer

    def evolve(self):
        self.population.sort(key=lambda x: x['score'], reverse=True)
        best = self.population[0]
        child = copy.deepcopy(best)
        child['params']['lr'] *= random.uniform(0.9, 1.1)
        child['score'] = 0
        self.population[-1] = child

def run_hivemind(args):
    print(f"\n>>> MODE: {args.mode.upper()} | Agents: {args.agents} | Generations: {args.generations}")

    env = UltimateLegionEnv(num_agents=args.agents, difficulty=0.0) # Start Easy
    hive = HiveMind()
    actions_map = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]

    if args.mode == 'train':
        print(f"{'Gen':<4} | {'Diff':<4} | {'Minerals':<8} | {'Surv%':<6} | {'Rho':<8}")
        print("-" * 50)

        for gen in range(args.generations):
            # Curriculum Update
            # Increase difficulty every 5 generations if performing well
            difficulty = min(1.0, gen / 20.0)
            env.set_difficulty(difficulty)

            for strategy in hive.population:
                policy, target, opt = hive.get_model(strategy['params'])
                if strategy['model_state']: policy.load_state_dict(strategy['model_state'])

                (vis, adj, cls) = env.reset()
                hidden = policy.init_hidden(args.agents)
                minerals = 0
                r_raw = 0

                for t in range(250):
                    with torch.no_grad():
                        q_vals, hidden = policy(vis, adj, cls, hidden)
                        actions = []
                        indices = []
                        eps = max(0.05, 0.5 * (0.9 ** gen))
                        for i in range(args.agents):
                            if random.random() < eps: idx = random.randint(0, 4)
                            else: idx = q_vals[i].argmax().item()
                            actions.append(actions_map[idx])
                            indices.append(idx)

                    (n_vis, n_adj, n_cls), r_raw, m = env.step(np.array(actions))
                    minerals += m

                    # Reward Normalization (Pop-Art)
                    hive.reward_normalizer.update(r_raw)
                    r_norm = hive.reward_normalizer.normalize(r_raw)

                    # Training
                    opt.zero_grad()
                    # Detach hidden state to prevent infinite backprop through time (BPTT) issues
                    # Ideally we use a sequence, but for online R-learning we truncate to 1 step
                    hidden_train = hidden.detach()

                    q_pred, _ = policy(vis, adj, cls, hidden_train)
                    q_act = q_pred[range(args.agents), indices]

                    with torch.no_grad():
                        q_next_vals, _ = target(n_vis, n_adj, n_cls, hidden_train)
                        q_next = q_next_vals.max(1)[0]

                    # Stabilized Target using Normalized Rewards
                    target_val = torch.FloatTensor(r_norm).to(device) - policy.rho + 0.95 * q_next
                    loss = F.mse_loss(q_act, target_val)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    opt.step()

                    # Adaptive Rho Update (on Normalized Rewards)
                    with torch.no_grad():
                        d = (torch.FloatTensor(r_norm).to(device) + 0.95*q_next - q_act).mean()
                        policy.rho += strategy['params']['rho_alpha'] * d
                        # No clamp needed because rewards are normalized N(0,1)!

                    vis, adj, cls = n_vis, n_adj, n_cls

                    if t % 5 == 0:
                        for tp, lp in zip(target.parameters(), policy.parameters()):
                            tp.data.copy_(0.1*lp.data + 0.9*tp.data)

                surv = np.mean(env.energy)
                strategy['score'] = minerals * 1000 + surv * 100
                strategy['stats'] = (minerals, surv, policy.rho.item())
                strategy['model_state'] = copy.deepcopy(policy.state_dict())

            best = max(hive.population, key=lambda x: x['score'])
            print(f"{gen+1:<4} | {difficulty:<4.2f} | {best['stats'][0]:<8.1f} | {best['stats'][1]:<6.2f} | {best['stats'][2]:<8.3f}")
            hive.evolve()

        torch.save(best['model_state'], args.save_path)
        print(f"Ultimate Brain Saved: {args.save_path}")

    # PLAY MODE
    elif args.mode == 'play':
        if not os.path.exists(args.save_path):
            print("Error: Brain file not found.")
            return

        print(">>> Loading Ultimate Brain...")
        model = UltimateLegionNet(grid_size=40).to(device)
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        model.eval()

        # Visualize Max Difficulty
        env.set_difficulty(1.0)
        (vis, adj, cls) = env.reset()
        hidden = model.init_hidden(args.agents)

        viz_pos, viz_adj, viz_cls = [], [], []

        print(">>> Running High-Res Simulation...")
        for t in range(400):
            viz_pos.append(env.positions.copy())
            viz_adj.append(adj.cpu().numpy())
            viz_cls.append(cls.cpu().numpy())

            with torch.no_grad():
                q_vals, hidden = model(vis, adj, cls, hidden)
                acts = [actions_map[i] for i in q_vals.argmax(1).cpu().numpy()]
            (vis, adj, cls), _, _ = env.step(np.array(acts))

        print(">>> Rendering GIF...")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 40); ax.set_ylim(0, 40)
        ax.set_facecolor('black')
        ax.set_title("Ultimate Legion: Memory + Telepathy + Physics")

        stars_x = [a['pos'][0] for a in env.asteroids]
        stars_y = [a['pos'][1] for a in env.asteroids]
        ax.plot(stars_x, stars_y, 'y*', markersize=4, alpha=0.3)

        scat = ax.scatter([], [], s=60)
        lines = []

        def update(i):
            pos = viz_pos[i]
            adj = viz_adj[i]
            cls = viz_cls[i]
            colors = ['cyan' if c == 0 else 'orange' for c in cls]
            scat.set_offsets(pos)
            scat.set_color(colors)

            [l.remove() for l in lines]
            lines.clear()

            count = 0
            for a in range(args.agents):
                for b in range(args.agents):
                    if a < b and adj[a,b] > 0 and count < 60:
                        l, = ax.plot([pos[a,0], pos[b,0]], [pos[a,1], pos[b,1]],
                                     color='lime', alpha=0.3, linewidth=0.5)
                        lines.append(l)
                        count += 1
            return scat, *lines

        ani = animation.FuncAnimation(fig, update, frames=len(viz_pos), interval=30)
        try:
            ani.save('outputs/ultimate_legion.gif', writer='pillow', fps=30)
            print("Visualization Saved: outputs/ultimate_legion.gif")
        except:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'])
    parser.add_argument('--agents', type=int, default=20)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--save_path', type=str, default='outputs/ultimate_brain.pth')

    args, _ = parser.parse_known_args()
    run_hivemind(args)