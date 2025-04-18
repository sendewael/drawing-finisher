import torch
from model import DrawingPredictor
import json
import os
import argparse

# dit zijn de parameters die ik kan meegeven in de cmd
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--steps', type=int, default=50, help='Number of points to predict')
    parser.add_argument('--output', type=str, default='completed_drawing.json',
                        help='Output JSON filename')
    parser.add_argument('--scale_max', type=float, default=400.0,
                        help='Normalization scale (must match training)')
    return parser.parse_args()

# laad het model uit het .pth bestand
# .eval zet het model in predictstand
def load_model(model_path: str, hidden_size: int = 64) -> DrawingPredictor:
    model = DrawingPredictor(hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def predict_and_save(
        model: DrawingPredictor,
        input_file: str,
        output_file: str,
        predict_steps: int = 50,
        sequence_length: int = 20,
        scale_max: float = 400.0
) -> None:

    # laadt de json met de halve circel
    with open(input_file, 'r') as f:
        data = json.load(f)

    # normaliseer de punten en ignore -1000's
    input_points = [(p['x'] / scale_max, p['y'] / scale_max) for p in data if p['x'] != -1000]

    # controleer of er genoeg punten zijn om een voorspelling te kunnen maken
    if len(input_points) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} points, got {len(input_points)}")

    predicted_points = []
    current_sequence = input_points[-sequence_length:]

    # per voorspelling: maak er een PyTorch tensor van
    # laat het model voorspellen
    # zet het resultaat terug om naar 0-400 waarden
    # schuif het inputvenster 1 plaats op
    for _ in range(predict_steps):
        input_tensor = torch.tensor([current_sequence], dtype=torch.float32)

        with torch.no_grad():
            pred = model(input_tensor).squeeze().tolist()

        denorm_pred = {'x': pred[0] * scale_max, 'y': pred[1] * scale_max}
        predicted_points.append(denorm_pred)

        current_sequence = current_sequence[1:] + [tuple(pred)]

    # plak het predicted resultaat aan het originele
    result = data + [{'x': -1000, 'y': -1000}] + predicted_points

    os.makedirs('output', exist_ok=True)
    output_path = os.path.join('output', output_file)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Prediction complete! Saved to {output_path}")
    print(f"Predicted {len(predicted_points)} points")

# dit voert alles uit
def main():
    args = parse_args()

    model = load_model("best_model.pth")
    predict_and_save(
        model=model,
        input_file=args.input,
        output_file=args.output,
        predict_steps=args.steps,
        scale_max=args.scale_max
    )

if __name__ == "__main__":
    main()