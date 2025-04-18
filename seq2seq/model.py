# model.py
import torch
import torch.nn as nn


class DrawingSeq2SeqPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_length=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_length = output_length

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        _, (hidden, cell) = self.encoder(input_seq)

        decoder_input = input_seq[:, -1].unsqueeze(1)
        outputs = []

        for _ in range(self.output_length):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            point = self.sigmoid(self.fc(out.squeeze(1)))
            outputs.append(point)
            decoder_input = point.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # Shape: [batch, output_length, 2]
