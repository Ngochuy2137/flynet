import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from python_utils.printer import Printer
from flynet.utils import utils as flynet_utils

# ---------------------------
# Configuration Class
# ---------------------------
class Config:
    def __init__(self, config_path='configs/training_config.json'):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        # Model params
        self.hidden_size = cfg['model']['hidden_size']
        self.num_layers = cfg['model']['num_layers']
        # Training params
        self.batch_size = cfg['training']['batch_size']
        self.lr = cfg['training']['learning_rate']
        self.num_epochs = cfg['training']['num_epochs']
        # Data sampling params
        self.num_data_files = cfg['data_sampling']['num_data_files']
        self.input_seg_len = cfg['data_sampling']['input_seg_len']
        self.random_sampling_params = cfg['data_sampling']['random_sampling_params']
        self.sche_sampling_params = cfg['data_sampling']['sche_sampling_params']


# ---------------------------
# Data Handling Class
# ---------------------------
class DataHandler:
    def __init__(self, data_parent_folder: str, config: Config, printer: Printer, interactive: bool = False):
        self.data_parent_folder = data_parent_folder
        self.config = config
        self.printer = printer
        self.interactive = interactive
        # List of object names (each should correspond to a folder and filename pattern)
        self.objects = [
            'ball', 'big_sized_plane', 'boomerang', 'cardboard', 'chip_star',
            'empty_bottle', 'empty_can', 'hat', 'rain_visor', 'ring_frisbee',
            'sand_can', 'soft_frisbee', 'basket', 'carpet', 'styrofoam'
        ]
        # Initialize container to hold the raw data
        self.input_data = {obj: [] for obj in self.objects}
        # Create a label mapping based on the order (or provide your own mapping)
        self.labels = {obj: idx for idx, obj in enumerate(self.objects)}
    
    def load_all_data(self):
        self.printer.print_blue('====================== LOADING CONFIG ======================', background=True)
        print(f'     Hidden size: {self.config.hidden_size}')
        print(f'     Number of layers: {self.config.num_layers}')
        print(f'     Batch size: {self.config.batch_size}')
        print(f'     Learning rate: {self.config.lr}')
        print(f'     Number of data files: {self.config.num_data_files}')
        print(f'     Random sampling parameters: {self.config.random_sampling_params}')
        print(f'     Scheduled sampling parameters: {self.config.sche_sampling_params}')
        
        if self.interactive:
            input('Press ENTER to continue ...')
        
        self.printer.print_blue('====================== LOADING DATA ======================', background=True)
        if self.interactive:
            input('Press ENTER to continue ...')
        
        for traj_idx in range(self.config.num_data_files):
            self.printer.print_blue(f'----- Trajectory ID: {traj_idx} -----')
            for obj in self.objects:
                file_path = os.path.join(self.data_parent_folder,
                                         obj,
                                         '3-data-augmented',
                                         'all',
                                         f'{obj}_{traj_idx}.csv')
                # Load data using your flynet utility. It should return an iterable of (sample, start_idx, file_path)
                data_iter = flynet_utils.load_data(
                    file_path,
                    time_steps=self.config.input_seg_len,
                    random_sampling_params=self.config.random_sampling_params,
                    sche_sampling_params=self.config.sche_sampling_params
                )
                for sample, start_idx, file_path in data_iter:
                    self.input_data[obj].append((sample, start_idx, file_path))
        
        self.printer.print_blue('Done loading data, starting setup input/label for training', background=True)
        if self.interactive:
            input('Press ENTER to continue ...')

    def prepare_datasets(self):
        # Combine data for each object into X and y arrays
        X_list, y_list = [], []
        for obj in self.objects:
            # Extract only the sample (first element in each tuple)
            traj_segments = [seg[0] for seg in self.input_data[obj]]
            X_obj = np.array(traj_segments)
            y_obj = np.full(len(X_obj), self.labels[obj])
            X_list.append(X_obj)
            y_list.append(y_obj)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print('X shape: ', X.shape)
        print('y shape: ', y.shape)
        
        # Split data into training (80%) and validation (20%)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape if needed for your LSTM input (Number of samples, Time steps, Features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
        
        # Create the dataset objects using your flynet_utils (assumes you have TimeSeriesDataset)
        train_dataset = flynet_utils.TimeSeriesDataset(X_train, y_train)
        val_dataset = flynet_utils.TimeSeriesDataset(X_val, y_val)
        return train_dataset, val_dataset, len(self.objects), X_train.shape[2]  # (train_dataset, val_dataset, num_classes, input_size)


# ---------------------------
# LSTM Model
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        # Use the last hidden state for classification
        out = self.fc(h_n[-1])
        return out


# ---------------------------
# Trainer Class
# ---------------------------
class Trainer:
    def __init__(self, config: Config, train_dataset, val_dataset, input_size, num_classes, device=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model, loss, and optimizer
        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=num_classes,
            num_layers=self.config.num_layers
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup directories and logging via wandb
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = os.path.join("models", current_time)
        os.makedirs(self.model_dir, exist_ok=True)
        self.wandb = flynet_utils.init_wandb(
            project_name='Flynet',
            run_name=f'hid{self.config.hidden_size}_layer{self.config.num_layers}_{current_time}',
            config={
                'input_size': input_size,
                'hidden_size': self.config.hidden_size,
                'output_size': num_classes,
                'num_layers': self.config.num_layers,
                'learning_rate': self.config.lr,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs
            },
            run_id=None, resume=None, wdb_notes=''
        )
        # Placeholders for tracking performance
        self.best_train_loss = float('inf')
        self.best_val_acc = 0.0
        self.losses = []
    
    def train(self):
        time_train_start = time.time()
        # Optionally, set up a live plot
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.losses.append(avg_epoch_loss)
            
            # Update loss plot
            ax.clear()
            ax.plot(range(1, len(self.losses) + 1), self.losses, linestyle='-')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid()
            if epoch > 100:
                ax.set_ylim(0, 0.1)
            plt.pause(0.1)
            
            print('-'*20)
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
            time_left = (time.time()-time_train_start) / (epoch+1) * (self.config.num_epochs - epoch - 1)
            print(f"    Time left: {time.strftime('%H:%M:%S', time.gmtime(time_left))}")
            
            # Validate model performance
            acc_rate = flynet_utils.evaluate_model(self.model, self.val_loader, self.device)
            print(f"Validation Accuracy: {acc_rate:.4f}")
            
            # Save model every 100 epochs
            if (epoch + 1) % 100 == 0:
                flynet_utils.save_model(self.model, self.optimizer, self.model_dir, epoch, losses=self.losses)
            
            # Save model if improved training loss
            if avg_epoch_loss < self.best_train_loss:
                self.best_train_loss = avg_epoch_loss
                flynet_utils.save_model(self.model, self.optimizer, self.model_dir, epoch, losses=self.losses, this_is_best_loss_model=self.losses)
            # Save model if improved validation accuracy
            if acc_rate > self.best_val_acc:
                self.best_val_acc = acc_rate
                flynet_utils.save_model(self.model, self.optimizer, self.model_dir, epoch, losses=self.losses, this_is_best_acc_model=acc_rate)
            
            self.wandb.log({
                'Avg Loss': avg_epoch_loss,
                'time left (min)': time_left/60,
                'Accuracy': acc_rate,
            }, step=epoch)
        
        plt.ioff()
        plt.show()


# ---------------------------
# Main Execution
# ---------------------------
def main():
    printer = Printer()  # Initialize your printer
    # Load configuration
    config = Config(config_path='configs/training_config.json')
    
    DATA_PARENT_FOLDER = os.getenv('NAE_DATASET20')
    # Create a DataHandler instance
    data_handler = DataHandler(DATA_PARENT_FOLDER, config, printer, interactive=False)
    data_handler.load_all_data()
    train_dataset, val_dataset, num_classes, input_size = data_handler.prepare_datasets()
    
    # Create the Trainer and start training
    trainer = Trainer(config, train_dataset, val_dataset, input_size, num_classes)
    trainer.train()


if __name__ == "__main__":
    main()
