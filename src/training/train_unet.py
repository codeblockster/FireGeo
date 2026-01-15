import torch
import torch.nn as nn

class UNetTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for data, target in train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
