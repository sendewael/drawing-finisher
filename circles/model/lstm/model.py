import torch.nn as nn
import torch

class DrawingPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, attention_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden]
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_output.mean(dim=1)  # simple mean pooling over sequence
        return self.fc(pooled)