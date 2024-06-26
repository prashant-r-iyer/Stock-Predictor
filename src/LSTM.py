import torch
import torch.nn as nn

# LSTM class
class LSTM(nn.Module):
    def __init__(self, input, hidden, stacked, device):
        super().__init__()

        self.hidden = hidden
        self.stacked = stacked
        self.device = device

        self.lstm = nn.LSTM(input, hidden, stacked, batch_first=True)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.stacked, batch_size, self.hidden).to(self.device)
        c0 = torch.zeros(self.stacked, batch_size, self.hidden).to(self.device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])

        return x
