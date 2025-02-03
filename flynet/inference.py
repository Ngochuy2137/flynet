import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from python_utils import filer

global_filer = filer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_PATH = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/flynet/flynet/models/20250202_171238/best_model.pth'
DATA_PARENT_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/flynet/flynet/objects'
# DATA_PARENT_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data'


# List of categories
categories = [
    'ball', 'big_sized_plane', 'boomerang', 'cardboard', 'chip_star',
    'empty_bottle', 'empty_can', 'hat', 'rain_visor', 'ring_frisbee',
    'sand_can', 'soft_frisbee', 'basket', 'carpet', 'styrofoam'
]

# Load data by randomly selecting 20 files from each category
def load_multiple_files(categories, num_files=20, num_samples_per_file=32, time_steps=30):
    input_data = {category: [] for category in categories}

    for category in categories:
        file_indices = np.random.randint(0, 100, num_files)  # Randomly select 20 files from 0 to 99
        file_paths = [f'{DATA_PARENT_DIR}/{category}/{category}_{i}.csv' for i in file_indices]
        # file_paths = global_filer.list_files(root_path=f'{DATA_PARENT_DIR}/{category}/3-data-augmented/data_plit/data_test_40', pattern='.*\.csv$')

        for file_path in file_paths:
            df = pd.read_csv(file_path, header=None)
            df = df.iloc[:, 1:]  # Remove the first column
            start_indices = np.random.randint(0, df.shape[0] - time_steps, num_samples_per_file)  # Select random start points

            for start_idx in start_indices:
                sample = df.iloc[start_idx:start_idx + time_steps, :].values.tolist()  # Extract data from random start point
                input_data[category].append(sample)

    return input_data

# Load data
num_files = 20
num_samples_per_file = 32
time_steps = 30
input_data = load_multiple_files(categories, num_files, num_samples_per_file, time_steps)

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Load the trained model
input_size = len(input_data['ball'][0][0])  # Get the number of features from the data
hidden_size = 64
output_size = len(categories)  # Number of classes

model = LSTMClassifier(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Convert test data
test_samples = []
test_labels = []
category_labels = {category: idx for idx, category in enumerate(categories)}

for category, samples in input_data.items():
    label = category_labels[category]
    for sample in samples:
        test_samples.append(sample)
        test_labels.append(label)

test_samples = np.array(test_samples, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.int64)

test_samples = torch.tensor(test_samples).to(device)
test_labels = torch.tensor(test_labels).to(device)

# Perform testing
correct = 0
total = len(test_samples)

with torch.no_grad():
    lstm_out, (h_n, c_n) = model.lstm(test_samples)  # Get LSTM output
    lstm_features = h_n[-1]  # Extract the last hidden state (input to FC layer)
    print('lstm_features shape: ', lstm_features.shape); input()
    # Print LSTM output before passing to FC layer
    print("LSTM output before FC layer (first 5 samples):")
    print(lstm_features[:5])  # Print the first 5 samples

    # Pass features through FC layer and perform classification
    outputs = model.fc(lstm_features)
    _, predicted = torch.max(outputs, 1)

    for i in range(len(predicted)):
        predicted_label = predicted[i].item()
        actual_label = test_labels[i].item()
        # print(f"Sample {i}: Predicted = {categories[predicted_label]}, Actual = {categories[actual_label]}")

        if predicted_label == actual_label:
            correct += 1

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")