import argparse
import os
import torch

from src.training import run_hivemind, run_aero, run_finance
from src.training_golden import train as train_golden, play as play_golden

# ==========================================
# 0. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing Simulation on: {device}")

if not os.path.exists('outputs'):
    os.makedirs('outputs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML simulations.")
    parser.add_argument('--experiment', type=str, default='legion', choices=['legion', 'aero', 'finance', 'legion-golden', 'all'],
                        help='The experiment to run.')

    # Arguments for the 'legion' (run_hivemind) experiment
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'], help='Legion mode: train or play.')
    parser.add_argument('--agents', type=int, default=20, help='Number of agents for Legion.')
    parser.add_argument('--generations', type=int, default=25, help='Number of generations to train for Legion.')
    parser.add_argument('--save_path', type=str, default='outputs/legion_brain_golden.pth', help='Path to save/load the Legion model.')
    parser.add_argument('--load_file', type=str, default='outputs/legion_brain_golden.pth', help='Path to load the Golden Legion model.')

    args, _ = parser.parse_known_args()

    if args.experiment == 'legion':
        print(f"Initializing HARDENED LEGION (Economic Stimulus) on: {device}")
        args.save_path = 'outputs/legion_brain_final.pth'
        run_hivemind(args)
    elif args.experiment == 'aero':
        run_aero()
    elif args.experiment == 'finance':
        run_finance()
    elif args.experiment == 'legion-golden':
        if args.mode == 'train':
            train_golden(args)
        else:
            play_golden(args)
    elif args.experiment == 'all':
        print("Running all experiments...")
        print(f"Initializing HARDENED LEGION (Economic Stimulus) on: {device}")
        args.save_path = 'outputs/legion_brain_final.pth'
        run_hivemind(args)
        run_aero()
        run_finance()
        print(f"Initializing LEGION GOLDEN EDITION on: {device}")
        args.save_path = 'outputs/legion_brain_golden.pth'
        train_golden(args)
