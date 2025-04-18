import torch
import json
import os
import argparse
from model import DrawingSeq2SeqPredictor

SCALE_MAX = 400.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--steps', type=int, default=30, help='Number of points to predict')
    parser.add_argument('--output', type=str, default='completed_drawing.json', help='Output filename')
    return parser.parse_args()


def load_input_points(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Normalize points and remove invalid ones
    points = [(p["x"] / SCALE_MAX, p["y"] / SCALE_MAX) for p in data if p["x"] != -1000]
    return points, data


def save_result(original_data, predicted_points, output_path):
    denormalized = [{'x': p[0] * SCALE_MAX, 'y': p[1] * SCALE_MAX} for p in predicted_points]
    result = original_data + [{'x': -1000, 'y': -1000}] + denormalized

    os.makedirs('output', exist_ok=True)
    with open(os.path.join('output', output_path), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Prediction saved to output/{output_path}")
    print(f"Predicted {len(predicted_points)} points")


def main():
    args = parse_args()

    model = DrawingSeq2SeqPredictor()
    model.load_state_dict(torch.load("best_seq2seq_model.pth", map_location='cpu'))
    model.eval()

    input_points, original_data = load_input_points(args.input)

    input_tensor = torch.tensor([input_points], dtype=torch.float32)  # shape: (1, input_len, 2)
    target_tensor = torch.zeros((1, args.steps, 2), dtype=torch.float32)

    with torch.no_grad():
        predicted = model(input_tensor)
  # shape: (1, steps, 2)
        predicted_points = predicted.squeeze(0).tolist()

    save_result(original_data, predicted_points, args.output)


if __name__ == "__main__":
    main()
