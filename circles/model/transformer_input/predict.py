import argparse
import torch
import json
import os
from model import CircleTransformer
from config import config
from utils import normalize_coordinates, denormalize_coordinates  # <-- using your utils!

def load_partial_circle(json_path):
    with open(json_path, 'r') as f:
        partial_circle = json.load(f)
    coords = torch.FloatTensor([[p['x'], p['y']] for p in partial_circle])
    coords = normalize_coordinates(coords)
    return coords.unsqueeze(0)  # (batch_size=1, seq_len, 2)

def save_completed_circle(coords, save_path):
    coords = denormalize_coordinates(coords.squeeze(0))
    coords = [{'x': float(x), 'y': float(y)} for x, y in coords.cpu().numpy()]
    with open(save_path, 'w') as f:
        json.dump(coords, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Predict missing points of a partial circle")
    parser.add_argument('--filename', type=str, required=True, help="Path to the partial circle JSON file")
    args = parser.parse_args()

    # Load model
    model = CircleTransformer()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()

    # Load partial circle
    partial_coords = load_partial_circle(args.filename)
    partial_coords = partial_coords.to(config.DEVICE)

    # Predict full circle
    predicted_coords = model.predict(partial_coords)

    # Save completed circle
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.filename))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_completed.json")
    save_completed_circle(predicted_coords, output_path)

    print(f"Completed circle saved to {output_path}")

if __name__ == "__main__":
    main()
