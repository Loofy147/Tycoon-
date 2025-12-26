import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from src.model import HardenedLegionNet, AeroNet, TraderNet
from src.environment import HardenedLegionEnv, AeroEnv, FinanceEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HiveMind:
    def __init__(self):
        self.population = []
        for _ in range(2):
            self.population.append({
                "params": {
                    "lr": 0.0003,
                    "rho_alpha": 0.005 # Slower Rho update for stability with large rewards
                },
                "score": 0, "model_state": None
            })

    def get_model(self, params):
        policy = HardenedLegionNet(grid_size=40).to(device)
        target = HardenedLegionNet(grid_size=40).to(device)
        target.load_state_dict(policy.state_dict())
        target.eval()
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

    env = HardenedLegionEnv(num_agents=args.agents, randomize=(args.mode=='train'))
    hive = HiveMind()
    actions_map = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]

    if args.mode == 'train':
        print(f"{'Gen':<4} | {'Minerals':<8} | {'Surv%':<6} | {'Rho':<8}")
        print("-" * 40)

        for gen in range(args.generations):
            for strategy in hive.population:
                policy, target, opt = hive.get_model(strategy['params'])
                if strategy['model_state']: policy.load_state_dict(strategy['model_state'])

                (vis, adj, cls) = env.reset()
                minerals = 0
                r_raw = 0

                for t in range(250):
                    with torch.no_grad():
                        q_vals = policy(vis, adj, cls)
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

                    opt.zero_grad()
                    q_act = policy(vis, adj, cls)[range(args.agents), indices]
                    with torch.no_grad():
                        q_next = target(n_vis, n_adj, n_cls).max(1)[0]

                    target_val = torch.FloatTensor(r_raw).to(device) - policy.rho + 0.95 * q_next
                    loss = F.mse_loss(q_act, target_val)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    opt.step()

                    # Exponential Dampening
                    with torch.no_grad():
                        current_rho_mag = abs(policy.rho.item())
                        adaptive_alpha = strategy['params']['rho_alpha'] * np.exp(-0.05 * current_rho_mag)
                        d = (torch.FloatTensor(r_raw).to(device) + 0.95*q_next - q_act).mean()
                        policy.rho += adaptive_alpha * d
                        policy.rho.clamp_(-50.0, 50.0) # WIDER CLAMP

                    vis, adj, cls = n_vis, n_adj, n_cls

                    if t % 5 == 0:
                        for tp, lp in zip(target.parameters(), policy.parameters()):
                            tp.data.copy_(0.1*lp.data + 0.9*tp.data)

                surv = np.mean(env.energy)
                strategy['score'] = minerals * 1000 + surv * 100
                strategy['stats'] = (minerals, surv, policy.rho.item())
                strategy['model_state'] = copy.deepcopy(policy.state_dict())

            best = max(hive.population, key=lambda x: x['score'])
            print(f"{gen+1:<4} | {best['stats'][0]:<8.1f} | {best['stats'][1]:<6.2f} | {best['stats'][2]:<8.3f}")
            hive.evolve()

        torch.save(best['model_state'], args.save_path)
        print(f"Verified Brain Saved: {args.save_path}")

    elif args.mode == 'play':
        if not os.path.exists(args.save_path):
            print("Error: Brain file not found. Train first!")
            return

        print(">>> Loading HiveMind Brain...")
        model = HardenedLegionNet(grid_size=40).to(device)
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        model.eval()

        viz_pos, viz_adj, viz_cls = [], [], []
        (vis, adj, cls) = env.reset()

        print(">>> Running High-Res Simulation...")
        for t in range(400):
            viz_pos.append(env.positions.copy())
            viz_adj.append(adj.cpu().numpy())
            viz_cls.append(cls.cpu().numpy())

            with torch.no_grad():
                q = model(vis, adj, cls)
                acts = [actions_map[i] for i in q.argmax(1).cpu().numpy()]
            (vis, adj, cls), _, _ = env.step(np.array(acts))

        print(">>> Rendering GIF...")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 40); ax.set_ylim(0, 40)
        ax.set_facecolor('black')
        ax.set_title("The Final Legion (Economic Balance)")

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
            ani.save('outputs/master_swarm.gif', writer='pillow', fps=30)
            print("Visualization Saved: outputs/master_swarm.gif")
        except Exception as e:
            print(f"Error saving animation: {e}")
            plt.show()

def run_aero(epochs=15):
    print("\n>>> LAUNCHING PROJECT AEOLUS...")
    env = AeroEnv(10)
    net = AeroNet().to(device)
    opt = optim.Adam(net.parameters(), lr=0.001)
    act_map = [(t,s) for t in [0,0.5,1] for s in [-1,0,1]]

    viz_pos = []
    for ep in range(epochs):
        s = env.reset()
        dist = 0
        for t in range(200):
            with torch.no_grad():
                q = net(s)
                a_idx = [q[i].argmax().item() if random.random()>0.1 else random.randint(0,8) for i in range(10)]

            act = np.array([act_map[i] for i in a_idx])
            ns, r = env.step(act)
            dist += np.mean(env.vel[:,0])

            opt.zero_grad()
            q_curr = net(s)[range(10), a_idx]
            with torch.no_grad(): q_next = net(ns).max(1)[0]
            td = torch.FloatTensor(r).to(device) - net.rho + 0.98*q_next
            F.mse_loss(q_curr, td).backward(); opt.step()

            with torch.no_grad(): net.rho += 0.01 * (torch.FloatTensor(r).to(device) + 0.98*q_next - q_curr).mean()
            s = ns
            if ep == epochs-1: viz_pos.append(env.pos.copy())

        print(f"Gen {ep+1}: Dist {dist:.1f} | Rho {net.rho.item():.3f}")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_xlim(0,50); ax.set_ylim(0,20)
    scat = ax.scatter([], [])
    def up(i): scat.set_offsets(viz_pos[i]); return scat,
    ani = animation.FuncAnimation(fig, up, frames=len(viz_pos))
    try:
        ani.save('outputs/aero.gif', writer='pillow', fps=30)
    except Exception as e:
        print(f"Failed to save animation: {e}")

def run_finance(epochs=15):
    print("\n>>> LAUNCHING WOLF ALGO...")
    env = FinanceEnv()
    net = TraderNet().to(device)
    opt = optim.Adam(net.parameters(), lr=0.001)

    hist = []
    for ep in range(epochs):
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q = net(obs)
                act = q.argmax(1).cpu().numpy()
                if random.random() < 0.1: act = np.random.randint(0,3,20)

            n_obs, r_scalar, val, done = env.step(act)

            opt.zero_grad()
            q_curr = net(obs)[range(20), act]
            with torch.no_grad(): q_next = net(n_obs).max(1)[0]

            r_tensor = torch.tensor(r_scalar, dtype=torch.float32, device=device)
            td = r_tensor - net.rho + 0.99*q_next

            F.mse_loss(q_curr, td).backward(); opt.step()

            with torch.no_grad(): net.rho += 0.01 * (r_tensor + 0.99*q_next - q_curr).mean()
            obs = n_obs

        print(f"Gen {ep+1}: Value ${val:.0f} | Rho {net.rho.item():.3f}")
        if ep == epochs-1: hist = env.hist

    plt.figure()
    plt.plot(hist, label='AI'); plt.legend(); plt.title('Portfolio')
    plt.savefig('outputs/finance.png')
