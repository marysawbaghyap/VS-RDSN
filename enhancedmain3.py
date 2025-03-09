from __future__ import print_function
import os
import random
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from torch.nn import functional as F
from rewards2 import compute_reward  # Updated reward function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
import vsum_tools


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
parser.add_argument('-d', '--dataset', type=str, required=True, help="Path to h5 dataset (required)")
parser.add_argument('-s', '--split', type=str, required=True, help="Path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="Split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'], help="Evaluation metric ['tvsum', 'summe']")
parser.add_argument('--resume', type=str, default='', help="Path to resume model checkpoint")
parser.add_argument('--evaluate', action='store_true', help="Evaluate model without training")
parser.add_argument('--gpu', type=str, default='0', help="Which GPU device to use")
parser.add_argument('--save-dir', type=str, default='log', help="Path to save outputs (default: 'log/')")
parser.add_argument('--verbose', action='store_true', help="Show detailed test results")
parser.add_argument('--save-results', action='store_true', help="Save output results")
parser.add_argument('--rnn-cell', type=str, default='lstm', choices=['rnn', 'lstm', 'gru'], help="RNN cell type (default: lstm)")
args = parser.parse_args()  # ✅ Ensure args is parsed globally before calling main()


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MotionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, motion_dim):
        super(MotionFeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_dim, motion_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class DSNWithSelfAttention(nn.Module):
    def __init__(self, in_dim=1024, motion_dim=64, scene_dim=128, hid_dim=256, num_heads=4):
        super(DSNWithSelfAttention, self).__init__()
        self.motion_extractor = MotionFeatureExtractor(in_dim, motion_dim)
        self.scene_extractor = nn.Linear(in_dim, scene_dim)
        
        # Self-Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=in_dim + motion_dim + scene_dim, num_heads=num_heads, batch_first=True)

        self.fc = nn.Linear(in_dim + motion_dim + scene_dim, 1)  # Importance score prediction

    def forward(self, x):
        motion_features = self.motion_extractor(x)
        scene_features = F.relu(self.scene_extractor(x))
        combined = torch.cat((x, motion_features, scene_features), dim=-1)

        # Self-Attention
        attn_output, _ = self.attention(combined, combined, combined)

        p = torch.sigmoid(self.fc(attn_output))  # Importance scores
        return p

# def main(args):
#     print("Initializing model with Self-Attention")
#     model = DSNWithSelfAttention(in_dim=1024, motion_dim=64, scene_dim=128, hid_dim=256, num_heads=4)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

#     if args.evaluate:
#         print("==> Evaluation Mode Activated")
#         if args.resume:
#             print(f"Loading checkpoint from {args.resume}")
#             checkpoint = torch.load(args.resume, map_location='cuda' if torch.cuda.is_available() else 'cpu')
#             model.load_state_dict(checkpoint)

#         model.eval()
#         dataset = h5py.File(args.dataset, 'r')
#         splits = read_json(args.split)
#         test_keys = splits[args.split_id]['test_keys']
#         evaluate(model, dataset, test_keys, torch.cuda.is_available())
#         return

#     # print("Training started")
#     # dataset = h5py.File(args.dataset, 'r')
#     # splits = read_json(args.split)
#     # train_keys = splits[args.split_id]['train_keys']
#     # print(train_keys)
#     # print(dataset[train_keys])
#     for epoch in range(10):
        
#         seq = torch.randn(1, 100, 1024)  # Example input
#         probs = model(seq)
#         m = Bernoulli(probs)
#         actions = m.sample()
#         reward = compute_reward(seq, actions, use_gpu=False, coverage_weight=0.3, stability_weight=0.2)
#         loss = -reward.mean()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch+1}, Reward: {reward.item():.4f}")


def main(args):
    print("Initializing model with Self-Attention")
    model = DSNWithSelfAttention(in_dim=1024, motion_dim=64, scene_dim=128, hid_dim=256, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # Set model to CPU explicitly
    device = torch.device("cpu")
    model.to(device)

    # Load dataset and training split
    dataset = h5py.File(args.dataset, 'r')
    splits = read_json(args.split)
    train_keys = splits[args.split_id]['train_keys']  # Get train video keys

    if args.evaluate:
        print("==> Evaluation Mode Activated")
        if args.resume:
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint)

        model.eval()
        dataset = h5py.File(args.dataset, 'r')
        splits = read_json(args.split)
        test_keys = splits[args.split_id]['test_keys']
        evaluate(model, dataset, test_keys, use_gpu=False)
        return  # ✅ Ensures training is skipped
    
    print("Training started")
    num_epochs = 10
    best_reward = float('-inf')  # Track the best reward
    best_epoch = -1
    model_save_path = osp.join(args.save_dir, 'best_model.pth.tar')

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_reward = 0.0

        for key in train_keys:
            seq = dataset[key]['features'][...]  # Load video features
            seq = torch.from_numpy(seq).unsqueeze(0).float().to(device)  # Convert to tensor (CPU)

            # Forward pass
            probs = model(seq)

            # Sample actions from Bernoulli distribution
            m = Bernoulli(probs)
            actions = m.sample()

            # Compute reward using actual video data
            reward = compute_reward(seq, actions, use_gpu=False, coverage_weight=0.6, stability_weight=0.2)

            # Compute loss (negative reinforcement learning reward)
            loss = -reward.mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and reward for logging
            total_loss += loss.item()
            total_reward += reward.mean().item()

        # Compute average loss and reward for the epoch
        avg_loss = total_loss / len(train_keys)
        avg_reward = total_reward / len(train_keys)

        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

        # Save model if the current epoch has the best reward
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch = epoch + 1
            save_checkpoint(model.state_dict(), model_save_path)
            print(f"✅ New best model saved at epoch {best_epoch} with reward {best_reward:.4f}")

    print("Training complete.")
    print(f"Best model achieved at epoch {best_epoch} with reward {best_reward:.4f}")
    print(f"Model saved at {model_save_path}")

    dataset.close()



def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")

    with torch.no_grad():
        model.eval()
        fms = []
        stability_scores = []
        diversity_scores = []
        coverage_scores = []

        for key in test_keys:
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]
            actions = torch.tensor(probs > 0.5, dtype=torch.float32)
            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'max')
            fms.append(fm)

            stability_scores.append(np.mean(np.diff(machine_summary)))
            diversity_scores.append(np.var(machine_summary))
            coverage_scores.append(np.mean(probs))

        if args.verbose:
            print("\n---  Video-Level F-scores  ---")
            print("No.\t\t\t Video\t\t\tF-score")
            for i, key in enumerate(test_keys):
                print(f"{i+1}\t\t\t{key}\t\t\t{fms[i]*100:.1f}%")
        print("----------------------------------")
        print(f"Average F-score {np.mean(fms) * 100:.1f}%")
        print(f"Stability Score: {np.mean(stability_scores):.4f}")
        print(f"Diversity Score: {np.mean(diversity_scores):.4f}")
        print(f"Coverage Score: {np.mean(coverage_scores):.4f}")
        mean_fm = np.mean(fms)
        print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

if __name__ == '__main__':
    main(args)
