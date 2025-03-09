from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
import random
from tabulate import tabulate
from torch.nn import functional as F
from rewards2 import compute_reward

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
# Set a fixed seed for stability
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

def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()  # Ensure model is in evaluation mode
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

            # Use deterministic frame selection to reduce variability
            actions = (probs > 0.5).float()

            machine_summary = vsum_tools.generate_summary(actions.numpy(), cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'max')
            fms.append(fm)

            stability_scores.append(np.mean(np.diff(machine_summary)))
            diversity_scores.append(np.var(machine_summary))
            coverage_scores.append(np.mean(probs))
        
        print(f"Average F-score: {np.mean(fms):.4f}")
        print(f"Stability Score: {np.mean(stability_scores):.4f}")
        print(f"Diversity Score: {np.mean(diversity_scores):.4f}")
        print(f"Coverage Score: {np.mean(coverage_scores):.4f}")

if __name__ == '__main__':
    def main(args):
        print("Initializing model")
        model = DSNWithMotionScene(in_dim=1024, motion_dim=64, scene_dim=128, hid_dim=256, num_layers=1, cell='lstm')
        print("Model initialized successfully")
    
    main(args)
