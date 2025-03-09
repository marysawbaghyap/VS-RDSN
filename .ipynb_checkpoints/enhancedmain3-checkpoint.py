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
from tabulate import tabulate
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
        x = F.relu(self.fc(x))
        return x

class DSNWithMotionScene(nn.Module):
    def __init__(self, in_dim=1024, motion_dim=64, scene_dim=128, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSNWithMotionScene, self).__init__()
        self.motion_extractor = MotionFeatureExtractor(in_dim, motion_dim)
        self.scene_extractor = nn.Linear(in_dim, scene_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim + motion_dim + scene_dim, hid_dim, num_layers=num_layers,
                               bidirectional=True, batch_first=True)
        elif cell == 'gru':
            self.rnn = nn.GRU(in_dim + motion_dim + scene_dim, hid_dim, num_layers=num_layers,
                             bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.RNN(in_dim + motion_dim + scene_dim, hid_dim, num_layers=num_layers,
                             bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        motion_features = self.motion_extractor(x)
        scene_features = F.relu(self.scene_extractor(x))
        combined = torch.cat((x, motion_features, scene_features), dim=-1)

        h, _ = self.rnn(combined)
        p = torch.sigmoid(self.fc(h))
        return p

def main(args):  # args is already passed, no need for "global args"
    print("Initializing model")
    model = DSNWithMotionScene(in_dim=1024, motion_dim=64, scene_dim=128, hid_dim=256, num_layers=1, cell='lstm')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    if args.evaluate:
        print("==> Evaluation Mode Activated")
        if args.resume:
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(checkpoint)

        model.eval()
        dataset = h5py.File(args.dataset, 'r')
        splits = read_json(args.split)
        test_keys = splits[args.split_id]['test_keys']
        evaluate(model, dataset, test_keys, torch.cuda.is_available())
        return  # ✅ Ensures training is skipped


    print("Training started")
    for epoch in range(10):  # Example epoch count
        seq = torch.randn(1, 100, 1024)  # Example input tensor
        probs = model(seq)

        m = Bernoulli(probs)
        actions = m.sample()
        reward = compute_reward(seq, actions, use_gpu=False, coverage_weight=0.3, stability_weight=0.2)
        
        loss = -reward.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Reward: {reward.item():.4f}")

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
            actions = torch.tensor(probs > 0.5, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor
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
                print(f"{i+1}\t\t\t{key}\t\t\t{fms[i]*100:.1f}%")  # Convert to percentage for readability
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
