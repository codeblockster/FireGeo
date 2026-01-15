import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# In real project this import would be:
# from src.models.pre_fire.lstm_model import FireRiskLSTM

class FireRiskTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in train_loader:
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
