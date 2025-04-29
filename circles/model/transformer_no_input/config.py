import torch

class Config:
    # Data parameters
    MAX_SEQ_LENGTH = 100  # Maximum number of points in a circle
    COORD_RANGE = (0, 400)  # Expected range of coordinates

    # Model parameters
    D_MODEL = 128  # Dimension of embeddings
    NHEAD = 8  # Number of attention heads
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    TRAIN_TEST_SPLIT = 0.8

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    TRAIN_DATA_PATH = '../../drawings/circles_training_data'
    MODEL_SAVE_PATH = 'circle_transformer_model.pth'


config = Config()
