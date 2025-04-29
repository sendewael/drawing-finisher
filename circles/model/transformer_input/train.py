import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import CircleTransformer
from data_loader import get_data_loaders
from config import config
from utils import save_model
import os
import time


def train_model():
    # Initialize lstm, optimizer, loss function
    model = CircleTransformer().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    # Get data loaders
    train_loader, test_loader = get_data_loaders(config.TRAIN_DATA_PATH)

    # Tensorboard writer
    writer = SummaryWriter()

    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            src = batch['src'].to(config.DEVICE)
            tgt_input = batch['tgt_input'].to(config.DEVICE)
            tgt_output = batch['tgt_output'].to(config.DEVICE)
            src_padding_mask = batch['src_padding_mask'].to(config.DEVICE)
            tgt_padding_mask = batch['tgt_padding_mask'].to(config.DEVICE)

            optimizer.zero_grad()

            output = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

            # Create loss mask for non-padded values
            # No unsqueeze, just ensure mask aligns with [batch_size, seq_len]
            loss_mask = ~tgt_padding_mask  # No need for unsqueeze if mask has shape [batch_size, seq_len]

            # Calculate loss only on non-padded parts
            # Use loss_mask to index the output and target tensors
            loss = criterion(output[loss_mask], tgt_output[loss_mask])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                src = batch['src'].to(config.DEVICE)
                tgt_input = batch['tgt_input'].to(config.DEVICE)
                tgt_output = batch['tgt_output'].to(config.DEVICE)
                src_padding_mask = batch['src_padding_mask'].to(config.DEVICE)
                tgt_padding_mask = batch['tgt_padding_mask'].to(config.DEVICE)

                output = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

                # Create loss mask for non-padded values
                loss_mask = ~tgt_padding_mask  # No need for unsqueeze here either
                loss = criterion(output[loss_mask], tgt_output[loss_mask])

                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        # Save best lstm
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            save_model(model, config.MODEL_SAVE_PATH)

        print(f'Epoch {epoch + 1}/{config.EPOCHS} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Test Loss: {avg_test_loss:.6f} | '
              f'Time: {time.time() - start_time:.2f}s')

    writer.close()


if __name__ == '__main__':
    train_model()
