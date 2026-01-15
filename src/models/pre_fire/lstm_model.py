import torch
import torch.nn as nn

class FireRiskLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return self.sigmoid(output)
