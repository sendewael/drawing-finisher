import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import config
from utils import normalize_coordinates, denormalize_coordinates

class CircleDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        random.shuffle(self.files)

        split_idx = int(len(self.files) * config.TRAIN_TEST_SPLIT)
        self.files = self.files[:split_idx] if split == 'train' else self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'r') as f:
            circle = json.load(f)

        # Convert to tensor and normalize
        coords = np.array([[point['x'], point['y']] for point in circle])
        coords = torch.FloatTensor(coords)
        coords = normalize_coordinates(coords)

        # Create mask for half circle
        seq_len = len(coords)
        half_len = seq_len // 2

        # Input is first half, target is full circle
        src = coords[:half_len]
        tgt = coords

        # Create padding masks
        src_padding_mask = torch.zeros(config.MAX_SEQ_LENGTH, dtype=torch.bool)
        tgt_padding_mask = torch.zeros(config.MAX_SEQ_LENGTH, dtype=torch.bool)

        src_padding_mask[half_len:] = True
        tgt_padding_mask[seq_len:] = True

        # For transformer, we need to shift the target for teacher forcing
        tgt_input = torch.cat([torch.zeros(1, 2), tgt[:-1]])

        return {
            'src': src,
            'tgt_input': tgt_input,
            'tgt_output': tgt,
            'src_padding_mask': src_padding_mask,
            'tgt_padding_mask': tgt_padding_mask
        }

def collate_fn(batch):
    # Pad sequences to MAX_SEQ_LENGTH
    padded_batch = {
        'src': torch.stack([torch.cat([item['src'],
                                       torch.zeros(config.MAX_SEQ_LENGTH - len(item['src']), 2)])
                            for item in batch]),
        'tgt_input': torch.stack([torch.cat([item['tgt_input'],
                                             torch.zeros(config.MAX_SEQ_LENGTH - len(item['tgt_input']), 2)])
                                  for item in batch]),
        'tgt_output': torch.stack([torch.cat([item['tgt_output'],
                                              torch.zeros(config.MAX_SEQ_LENGTH - len(item['tgt_output']), 2)])
                                   for item in batch]),
        'src_padding_mask': torch.stack([item['src_padding_mask'] for item in batch]),
        'tgt_padding_mask': torch.stack([item['tgt_padding_mask'] for item in batch])
    }
    return padded_batch

def get_data_loaders(data_dir):
    train_dataset = CircleDataset(data_dir, split='train')
    test_dataset = CircleDataset(data_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader
