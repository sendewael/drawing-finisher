import torch
import os
import json
from config import config

def normalize_coordinates(coords):
    """Normalize coordinates to [-1, 1] range"""
    min_val, max_val = config.COORD_RANGE
    coords = 2 * (coords - min_val) / (max_val - min_val) - 1
    return coords

def denormalize_coordinates(coords):
    """Convert normalized coordinates back to original range"""
    min_val, max_val = config.COORD_RANGE
    coords = (coords + 1) * (max_val - min_val) / 2 + min_val
    return coords

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_json_files(directory):
    """Load all JSON files from a directory"""
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data.append(json.load(f))
    return data