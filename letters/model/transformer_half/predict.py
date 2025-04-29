import json
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from model import LetterTransformer
from config import config
from utils import normalize_coordinates, denormalize_coordinates


def load_model(model_path):
    model = LetterTransformer().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    return model


def autoregressive_generate(model, total_points=50):
    """
    Auto-regressively generate points to create a Z shape from scratch.
    Handles the transformer's source-target requirement.
    """
    # Start with just one point (origin) for both src and tgt
    src = torch.FloatTensor([[0, 0]]).unsqueeze(0).to(config.DEVICE)  # (1, 1, 2)
    tgt = torch.FloatTensor([[0, 0]]).unsqueeze(0).to(config.DEVICE)  # (1, 1, 2)

    src = normalize_coordinates(src)
    tgt = normalize_coordinates(tgt)

    for _ in range(total_points - 1):
        output = model(src, tgt)  # Pass both src and tgt
        next_point = output[:, -1, :]  # Get last predicted point
        tgt = torch.cat([tgt, next_point.unsqueeze(1)], dim=1)  # Append to target

    generated = denormalize_coordinates(tgt.squeeze(0).detach().cpu())
    return [{'x': float(x), 'y': float(y)} for x, y in generated]


def generate_z(model_path=config.MODEL_SAVE_PATH, output_path='predicted/generated_z.json'):
    model = load_model(model_path)

    # Let the model generate the Z completely on its own
    generated_z = autoregressive_generate(model, total_points=50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(generated_z, f, indent=2)

    return generated_z


if __name__ == '__main__':
    predicted_z = generate_z()

    predicted_z_x = [p['x'] for p in predicted_z]
    predicted_z_y = [p['y'] for p in predicted_z]

    plt.figure(figsize=(8, 6))
    plt.plot(predicted_z_x, predicted_z_y, label='Generated Z', color='red', linewidth=2)
    plt.scatter(predicted_z_x, predicted_z_y, color='blue', s=10)  # Show individual points
    plt.gca().invert_yaxis()  # Invert y-axis to match typical coordinate system
    plt.legend()
    plt.title("Z Generated from Scratch by Model")
    plt.grid(True)
    plt.show()