# dataset.py
import os
import json
import torch
from torch.utils.data import Dataset


class Seq2SeqDrawingDataset(Dataset):
    def __init__(self, folder_path, input_length=50, target_length=50, scale_max=400.0):
        self.samples = []
        self.input_length = input_length
        self.target_length = target_length
        self.scale_max = scale_max

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)

                points = [(p["x"] / scale_max, p["y"] / scale_max)
                          for p in data if p["x"] != -1000]

                if len(points) >= input_length + target_length:
                    input_seq = points[:input_length]
                    target_seq = points[input_length:input_length + target_length]
                    self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(target_seq, dtype=torch.float32)
        )
