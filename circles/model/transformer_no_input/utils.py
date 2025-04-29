import torch

def normalize_coordinates(coords):
    # Normalize the coordinates within the range [0, 1]
    coords_min = coords.min(dim=0)[0]
    coords_max = coords.max(dim=0)[0]
    normalized_coords = (coords - coords_min) / (coords_max - coords_min)
    return normalized_coords

def denormalize_coordinates(coords):
    # Denormalize the coordinates back to the original range
    coords_min = torch.tensor([0, 0])  # This should be set to your actual min values
    coords_max = torch.tensor([400, 400])  # This should be set to your actual max values
    denormalized_coords = coords * (coords_max - coords_min) + coords_min
    return denormalized_coords
