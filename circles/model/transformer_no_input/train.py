import torch
import torch.optim as optim
from torch import nn
from model import CircleTransformer
from data_loader import get_data_loaders
from config import config

def train_model():
    # Load datasets
    train_loader, test_loader = get_data_loaders(config.TRAIN_DATA_PATH)

    # Initialize the lstm
    model = CircleTransformer().to(config.DEVICE)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(config.DEVICE)
            tgt_input = batch['tgt_input'].to(config.DEVICE)
            tgt_output = batch['tgt_output'].to(config.DEVICE)

            src_padding_mask = batch['src_padding_mask'].to(config.DEVICE)
            tgt_padding_mask = batch['tgt_padding_mask'].to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            output = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

            # Calculate loss
            loss = criterion(output, tgt_output)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{config.EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Average Loss: {total_loss / len(train_loader):.4f}")

        # Save the lstm after each epoch
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print("Training complete. Model saved at", config.MODEL_SAVE_PATH)


if __name__ == "__main__":
    train_model()
