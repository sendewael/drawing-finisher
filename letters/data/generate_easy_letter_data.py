import os
import json
import random
import numpy as np
from tqdm import tqdm

# Easier letters: mostly straight lines
LETTERS = ['L', 'I', 'V', 'T', 'Z']
NUM_SAMPLES_PER_LETTER = 4000
POINTS_PER_LETTER = 100
CANVAS_SIZE = 400

# Output folders
for letter in LETTERS:
    os.makedirs(f'letter_{letter.lower()}', exist_ok=True)

def base_shape(letter):
    shapes = {
        'L': [(0, 0), (0, 100), (50, 100)],
        'I': [(50, 0), (50, 100)],
        'V': [(0, 0), (50, 100), (100, 0)],
        'T': [(0, 0), (100, 0), (50, 0), (50, 100)],
        'Z': [(0, 0), (100, 0), (0, 100), (100, 100)],
    }
    return shapes.get(letter, [])

def transform(points, scale=1.0, rotation=0.0, tx=0, ty=0, jitter=0.0):
    transformed = []
    rad = np.radians(rotation)
    cos_r, sin_r = np.cos(rad), np.sin(rad)

    for x, y in points:
        # Scale
        x *= scale
        y *= scale
        # Rotate
        x_rot = x * cos_r - y * sin_r
        y_rot = x * sin_r + y * cos_r
        # Translate + jitter
        x_final = x_rot + tx + random.uniform(-jitter, jitter)
        y_final = y_rot + ty + random.uniform(-jitter, jitter)
        transformed.append((x_final, y_final))
    return transformed

def interpolate(points, num_points):
    if len(points) < 2:
        return [{"x": points[0][0], "y": points[0][1]}] * num_points
    resampled = []
    total_len = sum(np.linalg.norm(np.subtract(points[i], points[i+1])) for i in range(len(points)-1))
    dists = [0]
    for i in range(1, len(points)):
        dists.append(dists[-1] + np.linalg.norm(np.subtract(points[i], points[i-1])))

    interp_x = np.interp(np.linspace(0, dists[-1], num_points),
                         dists, [p[0] for p in points])
    interp_y = np.interp(np.linspace(0, dists[-1], num_points),
                         dists, [p[1] for p in points])
    return [{"x": float(x), "y": float(y)} for x, y in zip(interp_x, interp_y)]

# Generate JSON files
for letter in LETTERS:
    print(f"Generating '{letter}' samples...")
    base = base_shape(letter)

    for i in tqdm(range(NUM_SAMPLES_PER_LETTER)):
        scale = random.uniform(2.5, 3.5)
        rotation = random.uniform(-10, 10)  # Slight random rotation
        tx = random.uniform(100, 300)
        ty = random.uniform(100, 300)
        jitter = random.uniform(0.0, 3.0)   # Slight jitter

        shaped = transform(base, scale=scale, rotation=rotation, tx=tx, ty=ty, jitter=jitter)
        sampled = interpolate(shaped, POINTS_PER_LETTER)

        filename = f"letter_{letter.lower()}_{i:04d}.json"
        path = os.path.join(f'letter_{letter.lower()}', filename)
        with open(path, 'w') as f:
            json.dump(sampled, f, indent=2)

print("✅ Easier letter data generated!")
