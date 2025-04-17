# Project Overview
![Model architecture](Model%20architecture.png)
This repository contains scripts and models for training, evaluating, and visualizing results from LSTM-based models. Below is a brief description of each file:

## File Descriptions

### Scripts

#### `61_construct_dataset.py`
- This script is responsible for constructing the dataset from raw inputs.

#### `62_train_model.py`
- Trains the LSTM model using the preprocessed dataset created by `61_construct_dataset.py`.
- Outputs a trained model file (`LSTM_model_62.pt`) for later use in evaluation and visualization.

#### `72_train_model_change_loss.py`
- A modified version of the training script that uses a different loss function to evaluate its impact on model performance.

#### `74_plot_2d_compare.py`
- Generates 2D comparison plots for evaluating model predictions against ground truth values.
- Useful for analyzing model performance visually.

#### `75_plot_3_maps.py`
- Creates three-panel maps for visualizing predictions, ground truth, and their differences.
- Outputs high-resolution figures for further analysis.

---

### Model Files

#### `LSTM_model.pt`
- A version of the LSTM model trained using `72_train_model.py`.
- Used for predictions and further analysis.

#### `LSTM_model_62.pt`
- A version of the LSTM model trained using `62_train_model.py`.
