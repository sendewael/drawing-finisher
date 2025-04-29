import torch
from torch.utils.data import Dataset
import json
import os


class DrawingDataset(Dataset):
    def __init__(self, folder_path, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                points = [(p["x"], p["y"]) for p in data if p["x"] != -1000]

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