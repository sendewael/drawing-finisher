# Drawing Finisher 🎨✏️

![AI Drawing Completion](https://img.shields.io/badge/AI-Drawing%20Completion-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)

An AI system that completes partial or missing drawings by predicting the next points in a sequence.  
Developed as a proof-of-concept by Sen Dewael and Marnick Michielsen from Thomas More University (Geel, Belgium) in collaboration with KMITL (Bangkok, Thailand).

# Project Status 🚧

## Circles
- ✅ Transformer model: fully working — draws complete circles autonomously
- ⚙️ Other models (LSTM, etc.): functional but need code updates and testing

## Letters
- ⚙️ Models available but require more development and fixes

> **Note:** This project is a proof-of-concept (POC). Some models or features may be incomplete or experimental.

# Folder Structure 📂

```bash
root/
├── circles/
│   ├── data/
│   │   └── generate_full_circles.py
│   ├── models/
│   │   ├── lstm/
│   │   └── transformer_no_input/
├── letters/
│   ├── data/
│   │   └── generate_letter_data.py
│   ├── models/
│   │   └── transformer_input/
└── README.md
```

# Features ✨

- 🖌️ Complete partial drawings: **circles** and **capital letters A-E**
- 📚 Multiple model types: **Transformer**, **LSTM**, and others
- 🧠 Synthetic training data generation scripts
- 🔄 End-to-end workflow: from data generation to model prediction

# How It Works 🧠

## 1. Data Generation
Create synthetic training data for circles and letters.

## 2. Model Training
Train models on the generated datasets (Transformer model for circles currently works best).

## 3. Prediction
Feed a partial drawing or let the model generate full shapes.

## 4. Visualization (optional)
Visualize the generated outputs for evaluation.

# Getting Started 🚀

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/drawing-finisher.git
cd drawing-finisher
```

Intall the required packages:
```bash
pip install torch matplotlib numpy

```

## Generating Training Data

Navigate to the data folders and run the generation scripts.

For circles:
```bash
python circles/data/generate_full_circles.py
```

For letters:
```bash
python letters/data/generate_letter_data.py
```

## Training Models

Training scripts are organized inside each model folder.

Example: training the Transformer model for circles:
```bash
cd circles/models/transformer/
python train_transformer.py
```
> **Note:** Other models (LSTM, etc.) might need code updates to function correctly.

## Making Predictions

Use the prediction scripts provided inside the model folders.

Example: predicting a full circle with the Transformer model:
```bash
cd circles/models/transformer/
python predict_transformer.py
```
## Future Improvements 🔮

    Fully update and test LSTM and other models

    Add real-time drawing completion

    Improve letter prediction accuracy

    Create a web or GUI-based interface

## Contributors 👥

    Sen Dewael

    Marnick Michielsen
