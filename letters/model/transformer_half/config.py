import torch

class Config:
    # Data parameters
    MAX_SEQ_LENGTH = 100
    COORD_RANGE = (0, 400)

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
    TRAIN_DATA_PATH = '../../data/letters_training_data/letter_z'
    MODEL_SAVE_PATH = 'letter_transformer_model.pth'


config = Config()
