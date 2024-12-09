# Probability Policy Network

import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc5 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu((self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu((self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu((self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.relu((self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.softmax(self.fc5(x))
        return x

    
