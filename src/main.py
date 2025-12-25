import argparse
import os
import torch

from src.training import run_hivemind

# ==========================================
# 0. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing HARDENED LEGION (Economic Stimulus) on: {device}")

if not os.path.exists('outputs'):
    os.makedirs('outputs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'])
    parser.add_argument('--agents', type=int, default=20)
    parser.add_argument('--generations', type=int, default=25)
    parser.add_argument('--save_path', type=str, default='outputs/legion_brain_final.pth')

    args, _ = parser.parse_known_args()
    run_hivemind(args)