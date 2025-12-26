import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from src.model import UltimateLegionNet
from src.environment import GoldenLegionEnv
from src.utils import PopArtScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HiveMind:
    def __init__(self):
        self.pop = [{"params": {"lr": 0.0003, "rho_alpha": 0.01}, "score": 0, "model": None} for _ in range(2)]
        self.scaler = PopArtScaler()
    def get_model(self, p):
        pol = UltimateLegionNet(grid_size=40).to(device)
        tgt = copy.deepcopy(pol)
        opt = optim.Adam(pol.parameters(), lr=p['lr'])
        return pol, tgt, opt
    def evolve(self):
        self.pop.sort(key=lambda x: x['score'], reverse=True)
        self.pop[-1] = copy.deepcopy(self.pop[0])
        self.pop[-1]['params']['lr'] *= random.uniform(0.9, 1.1)

def train(args):
    print(f"\n>>> TRAINING LEGION (GOLDEN EDITION) | Agents: {args.agents} | Gen: {args.generations}")
    env = GoldenLegionEnv(args.agents)
    hive = HiveMind()
    act_map = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]

    print(f"{'Gen':<4} | {'Minerals':<8} | {'Surv%':<6} | {'Rho':<8}")
    print("-" * 40)

    for gen in range(args.generations):
        for strat in hive.pop:
            pol, tgt, opt = hive.get_model(strat['params'])
            if strat['model']: pol.load_state_dict(strat['model'])

            (vis, adj, cls) = env.reset()
            hid = pol.init_hidden(args.agents)
            mins = 0

            for t in range(250):
                with torch.no_grad():
                    q, hid = pol(vis, adj, cls, hid)
                    idx = [q[i].argmax().item() if random.random() > max(0.05, 0.5*0.9**gen) else random.randint(0,4) for i in range(args.agents)]

                (n_vis, n_adj, n_cls), r, m = env.step(np.array([act_map[i] for i in idx]))
                mins += m
                hive.scaler.update(r)
                r_norm = hive.scaler.normalize(r)

                opt.zero_grad()
                hid_d = hid.detach()
                q_act = pol(vis, adj, cls, hid_d)[0][range(args.agents), idx]

                with torch.no_grad():
                    q_next = tgt(n_vis, n_adj, n_cls, hid_d)[0].max(1)[0]

                td = torch.FloatTensor(r_norm).to(device) - pol.rho + 0.95*q_next
                F.mse_loss(q_act, td).backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step()

                with torch.no_grad():
                    mag = abs(pol.rho.item())
                    alpha = strat['params']['rho_alpha'] * np.exp(-0.1 * mag)
                    diff = (torch.FloatTensor(r_norm).to(device) + 0.95*q_next - q_act).mean()
                    pol.rho += alpha * diff
                    pol.rho.clamp_(-50.0, 50.0)

                vis, adj, cls = n_vis, n_adj, n_cls
                if t%5==0: tgt.load_state_dict(pol.state_dict())

            strat['score'] = mins * 1000 + np.mean(env.energy) * 100
            strat['stats'] = (mins, np.mean(env.energy), pol.rho.item())
            strat['model'] = copy.deepcopy(pol.state_dict())

        best = max(hive.pop, key=lambda x: x['score'])
        print(f"{gen+1:<4} | {best['stats'][0]:<8.1f} | {best['stats'][1]:<6.2f} | {best['stats'][2]:<8.3f}")
        hive.evolve()

    torch.save(best['model'], args.save_path)
    print(f"Legion Brain Saved: {args.save_path}")

def play(args):
    print(">>> RENDERING GOLDEN LEGION REPLAY...")
    env = GoldenLegionEnv(args.agents, randomize=True)
    model = UltimateLegionNet(grid_size=40).to(device)

    if not os.path.exists(args.load_file):
        print("Error: Brain file not found. Train first!")
        return

    model.load_state_dict(torch.load(args.load_file, map_location=device))
    act_map = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]

    viz_pos = []
    (vis, adj, cls) = env.reset()
    hid = model.init_hidden(args.agents)

    for _ in range(400):
        viz_pos.append(env.positions.copy())
        with torch.no_grad():
            q, hid = model(vis, adj, cls, hid)
            act = [act_map[i] for i in q.argmax(1).cpu().numpy()]
        (vis, adj, cls), _, _ = env.step(np.array(act))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('black')
    ax.set_xlim(0,40)
    ax.set_ylim(0,40)
    ax.set_title("The Golden Legion: High Reward Swarm")

    stars_x = [a['pos'][0] for a in env.asteroids]
    stars_y = [a['pos'][1] for a in env.asteroids]
    ax.plot(stars_x, stars_y, 'y*', markersize=4, alpha=0.3)

    scat = ax.scatter(env.positions[:, 0], env.positions[:, 1], c=['cyan' if c==0 else 'orange' for c in env.classes], s=50)

    def up(i):
        scat.set_offsets(viz_pos[i])
        return scat,

    ani = animation.FuncAnimation(fig, up, frames=len(viz_pos))
    try:
        out = args.load_file.replace('.pth', '.gif')
        ani.save(out, writer='pillow', fps=30)
        print(f"GIF Saved: {out}")
    except:
        plt.show()
