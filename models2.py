import torch
import torch.nn as nn
import torch.nn.functional as F

class DSN(nn.Module):
    """Deep Summarization Network with CNN feature extraction and Attention mechanism."""
    def __init__(self, in_channels, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru', 'rnn'], "cell must be either 'lstm', 'gru', or 'rnn'"
        
        # CNN for feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # RNN part
        rnn_input_dim = 128 * (in_dim // 4) # Adjust based on your pooling and conv layers
        if cell == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        elif cell == 'gru':
            self.rnn = nn.GRU(rnn_input_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.RNN(rnn_input_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        
        # Attention Layer
        self.attention = nn.Linear(hid_dim * 2, 1)
        self.fc = nn.Linear(hid_dim * 2, 1)
    
    def forward(self, x):
        # Apply conv layers + pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the features for the RNN
        x = x.view(x.size(0), -1)
        
        # RNN output
        h, _ = self.rnn(x)
        
        # Apply attention
        attn_weights = torch.softmax(self.attention(h), dim=1)
        h = h * attn_weights
        
        # Final classification
        p = torch.sigmoid(self.fc(h))
        return p
