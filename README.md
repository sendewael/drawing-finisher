# Drawing Finisher 🎨✏️

![AI Drawing Completion](https://img.shields.io/badge/AI-Drawing%20Completion-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)

An AI model that completes partial drawings by predicting the next points in a sequence. Developed as a proof-of-concept by Sen Dewael and Marnick Michielsen from Thomas More University (Geel, Belgium) in collaboration with KMITL (Bangkok, Thailand). 

#### Keep in mind that this AI model is POC which means that the featues are not perfectly worked out and the AI model needs work to create perfect predictions.

## Features ✨

- 🖌️ Completes partial drawings (circles and capital letters A-E)
- 📈 LSTM-based neural network for sequential prediction
- 🧠 Trained on synthetically generated drawing data
- 📊 Includes data visualization tools
- 🔄 End-to-end pipeline from data generation to prediction

## How It Works 🧠

The system processes drawing sequences through these stages:

1. **Data Generation**: Synthetic training data is created for circles and letters
2. **Data Processing**: Coordinates are normalized and sequenced
3. **Model Training**: LSTM network learns drawing patterns
4. **Prediction**: Model completes partial drawings
5. **Visualization**: Results can be viewed as plotted drawings

## Getting Started 🚀

### Installation

1. Clone the repository
2. Install required packages:
   - PyTorch
   - Matplotlib
   - NumPy

### Generating Training Data

Navigate to the drawings folder and run:
- `generate_full_circles.py` for circle data
- `generate_letter_data.py` for letter data (A-E)

### Training the Model

Training script:
- `train.py`

Execute the training script which will:
- Load and preprocess the drawing data
- Train the LSTM model
- Save the best performing model
- Display training progress metrics

### Making Predictions

Prediction script:
- `predict.py --input <filename> --steps <number of points to predict> --output <output_filename>`

Use the prediction script with:
- An input JSON file containing partial drawing
- Number of points to predict
- Output file name for completed drawing

### Visualizing Results

Visualizing script:
- `visualize_drawing.py <filename>`

The visualization tool can display:
- Original partial drawings
- Model-completed drawings
- Comparison between input and output

## Model Architecture 🏗️

| Component       | Specification              |
|-----------------|----------------------------|
| Model Type      | LSTM Neural Network        |
| Input Size      | 2 (x,y coordinates)        |
| Hidden Size     | 64 units                   |
| Output Size     | 2 (next x,y prediction)    |
| Activation      | Sigmoid (output layer)     |
| Loss Function   | Mean Squared Error         |
| Optimizer       | Adam                       |

## Performance Metrics 📊

Typical training results show:

| Metric          | Training | Validation |
|-----------------|----------|------------|
| Normalized Loss | 0.000123 | 0.000135   |
| Raw Loss        | 19.7     | 21.6       |

## Example Workflow 🔄

1. Prepare partial drawing as JSON file
2. Run prediction to generate additional points
3. Save completed drawing as new JSON
4. Visualize results to verify quality

## Troubleshooting 🛠️

**Common Issues:**

- **Insufficient input points**: Ensure at least 10 coordinate points
- **Coordinate range**: Input values should be 0-400 scale
- **File paths**: Use correct relative paths when running scripts

## Future Improvements 🔮

- Support for free-form drawings
- Real-time prediction capabilities
- Web-based interface
- Enhanced visualization tools

## Contributors 👥

- **Sen Dewael**
- **Marnick Michielsen**

Developed by students of Thomas More University of Applied Sciences, Geel, Belgium in collaboration with KMITL, Bangkok, Thailand.
