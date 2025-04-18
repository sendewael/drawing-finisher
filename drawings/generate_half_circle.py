import json
import math

def generate_half_circle_points(radius=50, center=(200, 200), num_points=30):
    points = []
    for i in range(num_points):
        angle = math.pi * (i / num_points)  # Half circle (0 to π)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append({"x": x, "y": y})
    return points

half_circle = generate_half_circle_points()

# Save it
with open("../model/half_circle_input.json", "w") as f:
    json.dump(half_circle, f, indent=2)

print("🌓 Half-circle saved as 'half_circle_input.json'")
