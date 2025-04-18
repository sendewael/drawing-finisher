import torch
from torch.utils.data import Dataset
import json
import os

# om model te trainen op eigen data met PyTorch moeten we Dataset gebruiken
class DrawingDataset(Dataset):
    def __init__(self, folder_path, sequence_length=20):
        self.sequence_length = sequence_length
        self.samples = []
        # loop door alle jsons, en haal de x, y coordinaten er uit. Ignore de -1000's.
        # maak tuples van x, y en plaats in data
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                points = [(p["x"], p["y"]) for p in data if p["x"] != -1000]

                # hier worden pairs gemaakt van input en target. sequence length is 10, dus voorspel 11de punt
                # plaats deze dan in samples
                for i in range(len(points) - self.sequence_length):
                    input_seq = points[i:i + self.sequence_length]
                    target = points[i + self.sequence_length]
                    self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]

        # Convert to tensor
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return input_tensor, target_tensor
