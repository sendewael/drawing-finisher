import torch.nn as nn

class DrawingPredictor(nn.Module):
    # input size = elk punt heeft 2 values: x en y
    # hidden size = grootte van LSTM memory
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        # fully connected layer
        # outputs 2 nummers: x en y
        # sigmoid zet nummers tussen de 0 en 1 (normalizen)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])