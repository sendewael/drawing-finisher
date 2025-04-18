import json
import os
import math
import random
from tqdm import tqdm

# Where to save generated circles
output_dir = "circles_output"
os.makedirs(output_dir, exist_ok=True)

# Configuration
NUM_CIRCLES = 4000
POINTS_PER_CIRCLE = 100
SCALE_MAX = 400.0  # Match training scale

def generate_circle_json(index):
    # Random center and radius
    radius = random.uniform(40, 120)
    cx = random.uniform(radius + 10, SCALE_MAX - radius - 10)
    cy = random.uniform(radius + 10, SCALE_MAX - radius - 10)

    # Random start angle (0 to 2π)
    start_angle = random.uniform(0, 2 * math.pi)

    # Random direction
    clockwise = random.choice([True, False])

    # Optional jitter (to avoid perfect circles)
    jitter_strength = random.uniform(0.0, 2.5)

    points = []
    for i in range(POINTS_PER_CIRCLE):
        t = i / POINTS_PER_CIRCLE
        angle = start_angle + (t * 2 * math.pi * (-1 if clockwise else 1))
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        # Add slight noise to x, y for realism
        x += random.uniform(-jitter_strength, jitter_strength)
        y += random.uniform(-jitter_strength, jitter_strength)

        points.append({"x": x, "y": y})

    # Save as JSON
    filename = f"circle_{index:04d}.json"
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(points, f, indent=2)

# Generate all circles
print(f"Generating {NUM_CIRCLES} circle JSONs in '{output_dir}'...")
for i in tqdm(range(NUM_CIRCLES)):
    generate_circle_json(i)

print("✅ Done! You can now train your model on this dataset.")
