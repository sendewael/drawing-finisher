# predict zelf zonder input
import json
import torch
from model import CircleTransformer
from config import config
from utils import normalize_coordinates, denormalize_coordinates
import numpy as np
import random
import math


def load_model(model_path):
    model = CircleTransformer().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    return model


def generate_half_circle(num_points=50, radius=100, center=(200, 200)):
    """
    Generates a half-circle by defining a set of points along the top half of a circle.

    :param num_points: Number of points to generate on the half-circle.
    :param radius: The radius of the circle.
    :param center: The (x, y) coordinates of the circle center.
    :return: A list of points on the half-circle.
    """
    half_circle_points = []
    start_angle = math.pi  # Start from -π (left side of the circle)
    end_angle = 2 * math.pi  # End at π (right side of the circle)

    for i in range(num_points):
        angle = start_angle + (i / (num_points - 1)) * (end_angle - start_angle)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        half_circle_points.append({'x': x, 'y': y})

    return half_circle_points


def predict_full_circle(model, half_circle):
    # Assuming half_circle is normalized and in the correct format (list of points)
    coords = torch.FloatTensor([[point['x'], point['y']] for point in half_circle])
    coords = normalize_coordinates(coords).to(config.DEVICE)

    # Get the predicted full circle
    completed_circle = model.predict(coords)

    # Denormalize and convert to list of points
    completed_circle = denormalize_coordinates(completed_circle.cpu())
    completed_points = [{'x': float(x), 'y': float(y)} for x, y in completed_circle]

    return completed_points


def complete_circle(model_path=config.MODEL_SAVE_PATH, output_path='predicted/completed_circle.json'):
    # Load lstm
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.")

    # Generate random half circle
    print("Generating random half-circle...")
    half_circle = generate_half_circle(num_points=50, radius=100, center=(200, 200))
    print(f"Half circle generated. {len(half_circle)} points.")

    # Predict the full circle by completing the half-circle
    print("Predicting full circle...")
    completed_circle = predict_full_circle(model, half_circle)
    print(f"Prediction complete. {len(completed_circle)} points predicted.")

    # Save the completed circle
    print(f"Saving completed circle to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(completed_circle, f, indent=2)
    print(f"Completed circle saved to {output_path}")
    return completed_circle


if __name__ == '__main__':
    complete_circle()  # No need to provide an input file anymore
