import torch
from torch.utils.data import DataLoader, random_split
from model import DrawingPredictor
from dataset import DrawingDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# parameters
EPOCHS = 30 # hoeveel keer ziet het model de volledige training set
BATCH_SIZE = 64 # hoeveel samples geven we het model per keer
LEARNING_RATE = 0.0001 # learning steps
SEQUENCE_LENGTH = 20 # hoeveel punten ziet het model vooraleer het volgende te voorspellen
VALIDATION_RATIO = 0.2 # hoeveel procent van de data om te evalueren
SCALE_MAX = 400.0 # normalizeren, 0-400 wordt 0-1

# zet alle coordinaten van de dataset om naar 0-1
class NormalizedDrawingDataset(DrawingDataset):
    def __getitem__(self, idx):
        input_seq, target = super().__getitem__(idx)
        return input_seq / SCALE_MAX, target / SCALE_MAX

# laad en normalizeer de volledige dataset
full_dataset = NormalizedDrawingDataset(
    folder_path="../drawings/circles_output",
    sequence_length=SEQUENCE_LENGTH
)

# split de dataset in train en valideer
train_size = int((1 - VALIDATION_RATIO) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# wrap datasets zodat PyTorch batches van data kan laden tijdens trainen
# shuffle = true, zorgt ervoor dat trainings data gemixed wordt voor beter trainen
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# laatste layer van het model, 2 input points en output tussen 0-1 zetten
class StableDrawingPredictor(DrawingPredictor):
    def __init__(self):
        super().__init__(input_size=2, hidden_size=64)
        self.fc = nn.Sequential(
            nn.Linear(64, 2),
            nn.Sigmoid()
        )


model = StableDrawingPredictor() # LSTM model
criterion = nn.MSELoss() # loss functie
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # adam algorithme om weights te optimizen
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2) # automatisch learning rate aanpassen als progress verminderd

# loop door epochs
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        optimizer.zero_grad() # reset gradients
        outputs = model(inputs) # predict output
        loss = criterion(outputs, targets) # bereken loss
        loss.backward() # backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
        optimizer.step() # update weights
        train_loss += loss.item()

    # loop door evaluation data, bereken de loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch + 1}:")
    print(f"  Norm Loss: {train_loss:.6f} (Train), {val_loss:.6f} (Val)")
    print(f"  Raw Loss: {train_loss * SCALE_MAX ** 2:.1f} (Train), {val_loss * SCALE_MAX ** 2:.1f} (Val)")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # als het huidige model beter is dan het vorige, save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("  🎯 New best model saved!")

    # pas de learning rate aan
    scheduler.step(val_loss)

print(
    f"\nTraining complete. Best validation loss: {best_val_loss:.6f} (norm), {best_val_loss * SCALE_MAX ** 2:.1f} (raw)")