# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Seq2SeqDrawingDataset
from model import DrawingSeq2SeqPredictor

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
INPUT_LENGTH = 50
TARGET_LENGTH = 50
SCALE_MAX = 400.0

dataset = Seq2SeqDrawingDataset(
    folder_path="../drawings/circles_output",
    input_length=INPUT_LENGTH,
    target_length=TARGET_LENGTH,
    scale_max=SCALE_MAX
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DrawingSeq2SeqPredictor(
    input_size=2,
    hidden_size=64,
    output_length=TARGET_LENGTH
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.6f}")

torch.save(model.state_dict(), "best_seq2seq_model.pth")
print("✅ Model saved as best_seq2seq_model.pth")
