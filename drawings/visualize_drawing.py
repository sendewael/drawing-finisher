import json
import matplotlib.pyplot as plt
import sys
import os

def visualize_drawing(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    x, y = [], []
    plt.figure(figsize=(6, 6))

    for point in data:
        if point["x"] == -1000 and point["y"] == -1000:
            # Draw current line and start a new one
            if x and y:
                plt.plot(x, y, marker='o')
                x, y = [], []
        else:
            x.append(point["x"])
            y.append(point["y"])

    # Draw the last segment
    if x and y:
        plt.plot(x, y, marker='o')

    plt.gca().invert_yaxis()  # To match your web canvas coordinate system
    plt.title(os.path.basename(file_path))
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_drawing.py path/to/file.json")
    else:
        visualize_drawing(sys.argv[1])
