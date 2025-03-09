import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torchvision import models

#__all__ = ['DSN']

__all__ = ['DSN', 'FNN', 'Transformer']
class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru', 'rnn'], "cell must be either 'lstm', 'gru', or 'rnn'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        elif cell == 'gru':
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:  
            self.rnn = nn.RNN(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = torch.sigmoid(self.fc(h))
        return p



class Transformer(nn.Module):
    """Transformer-based Model"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1):
        super(Transformer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=in_dim, nhead=8, dim_feedforward=hid_dim)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.sigmoid(self.fc(x))
        return x

class FNN(nn.Module):
    """Feedforward Neural Network (FNN)"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class MotionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, motion_dim):
        super(MotionFeatureExtractor, self).__init__()
        # Define your motion feature extraction layers here
        self.conv1 = nn.Conv2d(input_dim, motion_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Compute motion features
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x

class DSNWithMotion(nn.Module):
    def __init__(self, in_dim=1024, motion_dim=64, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSNWithMotion, self).__init__()
        self.motion_extractor = MotionFeatureExtractor(in_dim, motion_dim)
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim + motion_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        elif cell == 'gru':
            self.rnn = nn.GRU(in_dim + motion_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:  
            self.rnn = nn.RNN(in_dim + motion_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        motion_features = self.motion_extractor(x)
        x = torch.cat((x, motion_features), dim=1)
        h, _ = self.rnn(x)
        p = torch.sigmoid(self.fc(h))
        return p