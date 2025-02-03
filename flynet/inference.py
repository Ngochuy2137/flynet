import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from python_utils import filer

class DataLoader:
    def __init__(self, data_dir, categories, num_files=20, num_samples_per_file=32, time_steps=30):
        self.data_dir = data_dir
        self.categories = categories
        self.num_files = num_files
        self.num_samples_per_file = num_samples_per_file
        self.time_steps = time_steps

    def load_data(self):
        input_data = {category: [] for category in self.categories}

        for category in self.categories:
            file_indices = np.random.randint(0, 100, self.num_files)
            file_paths = [f'{self.data_dir}/{category}/{category}_{i}.csv' for i in file_indices]

            for file_path in file_paths:
                df = pd.read_csv(file_path, header=None)
                df = df.iloc[:, 1:]
                start_indices = np.random.randint(0, df.shape[0] - self.time_steps, self.num_samples_per_file)

                for start_idx in start_indices:
                    sample = df.iloc[start_idx:start_idx + self.time_steps, :].values.tolist()
                    input_data[category].append(sample)

        return input_data


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


class ModelEvaluator:
    def __init__(self, categories):
        self.categories = categories
        self.category_labels = {category: idx for idx, category in enumerate(categories)}

    def prepare_data(self, input_data):
        test_samples = []
        test_labels = []

        for category, samples in input_data.items():
            label = self.category_labels[category]
            for sample in samples:
                test_samples.append(sample)
                test_labels.append(label)

        test_samples = np.array(test_samples, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.int64)

        return torch.tensor(test_samples), torch.tensor(test_labels)

    def evaluate(self, predictions, test_labels, device):
        correct = (predictions.to(device) == test_labels.to(device)).sum().item()
        accuracy = 100 * correct / len(test_labels)
        return accuracy


class FlyNetInfer:
    def __init__(self, model_path, output_size, input_size=3, hidden_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.load_model(input_size, output_size, hidden_size)

    def load_model(self, input_size, output_size, hidden_size):
        print('FLYNET: Loading model from', self.model_path)
        self.model = LSTMClassifier(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def extract(self, test_samples, get_feature_only=False):
        '''
        extract feature of trajectory data
        '''
        # convert to torch tensor if not
        if not isinstance(test_samples, torch.Tensor):
            test_samples = torch.tensor(test_samples, dtype=torch.float32)
        with torch.no_grad():
            lstm_out, (h_n, c_n) = self.model.lstm(test_samples.to(self.device))
            lstm_features = h_n[-1]
            # print('lstm_features shape: ', lstm_features.shape)
            if get_feature_only:
                return lstm_features
            
            print("LSTM output before FC layer (first 5 samples):")
            # print(lstm_features[:5])
            outputs = self.model.fc(lstm_features)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def offline_run(self, data_parent_dir, categories):
        data_loader = DataLoader(data_parent_dir, categories)
        input_data = data_loader.load_data()
        evaluator = ModelEvaluator(categories)
        test_samples, test_labels = evaluator.prepare_data(input_data)
        predictions = self.extract(test_samples)
        accuracy = evaluator.evaluate(predictions, test_labels, self.device)

        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    MODEL_PATH = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/flynet/flynet/models/20250202_171238/best_model.pth'
    DATA_PARENT_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/flynet/flynet/objects'
    categories = [
        'ball', 'big_sized_plane', 'boomerang', 'cardboard', 'chip_star',
        'empty_bottle', 'empty_can', 'hat', 'rain_visor', 'ring_frisbee',
        'sand_can', 'soft_frisbee', 'basket', 'carpet', 'styrofoam'
    ]

    system = FlyNetInfer(MODEL_PATH, input_size=3, output_size=len(categories), hidden_size=64)
    system.offline_run(data_parent_dir=DATA_PARENT_DIR, categories=categories)