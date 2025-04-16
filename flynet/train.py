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
from nae_static.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae_static.utils.submodules.training_utils.input_label_generator import InputLabelGenerator

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
class DataLoaderFlynet:
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
        # self.input_data = {obj: [] for obj in self.objects}
        # Create a label mapping based on the order (or provide your own mapping)
        self.labels = {obj: idx for idx, obj in enumerate(self.objects)}
    
    def load_all_train_val_test_dataset(self, data_train_lim=None, data_val_lim=None):
        '''
        Load all data of the specified objects from the data parent folder.
        '''
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
        
        # Load from data_train
        X_train_list = []
        y_train_list = []
        # load from data_val, data_test
        X_val_list = []
        y_val_list = []
        nae_data_loader = NAEDataLoader() 
        input_label_generator = InputLabelGenerator()

        for obj in self.objects:
            data_split_path = os.path.join(self.data_parent_folder,
                                        obj,
                                        '3-data-augmented',
                                        'data_plit')
            data_train, data_val, data_test = nae_data_loader.load_train_val_test_dataset(data_split_path, file_format='csv')
            data_val = data_val + data_test

            if data_train_lim is not None:
                data_train = data_train[:data_train_lim]
            if data_val_lim is not None:
                data_val = data_val[:data_val_lim]

            data_train  = input_label_generator.generate_input_label_static_seq(data_train, self.config.input_seg_len, 10)
            X_train = data_train[0]  # only use input_seq element
            data_val    = input_label_generator.generate_input_label_static_seq(data_val, self.config.input_seg_len, 10)
            X_val = data_val[0]  # only use input_seq element
            
            y_train = np.full(len(X_train), self.labels[obj])
            y_val = np.full(len(X_val), self.labels[obj])

            # Append to the list
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_val_list.append(X_val)
            y_val_list.append(y_val)
            self.printer.print_blue(f'Loaded {len(X_train)} training samples and {len(X_val)} validation samples for {obj}', background=True)
            if self.interactive:
                input('Press ENTER to continue ...')

        self.printer.print_blue('Done loading data, starting setup input/label for training', background=True)
        if self.interactive:
            input('Press ENTER to continue ...')
        
        X_train_all = np.concatenate(X_train_list, axis=0)
        y_train_all = np.concatenate(y_train_list, axis=0)
        X_val_all = np.concatenate(X_val_list, axis=0)
        y_val_all = np.concatenate(y_val_list, axis=0)

        train_dataset = flynet_utils.TimeSeriesDataset(X_train_all, y_train_all)
        val_dataset = flynet_utils.TimeSeriesDataset(X_val_all, y_val_all)

        return train_dataset, val_dataset, len(self.objects), X_train.shape[2]  # (train_dataset, val_dataset, num_classes, feature_dim)


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
    def __init__(self, config: Config, train_dataset, val_dataset, feature_dim, num_classes, device=None, enable_wandb=False):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model, loss, and optimizer
        self.model = LSTMClassifier(
            input_size=feature_dim,
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

        self.enable_wandb = enable_wandb
        if self.enable_wandb:
            self.wandb = flynet_utils.init_wandb(
                project_name='Flynet',
                run_name=f'hid{self.config.hidden_size}_layer{self.config.num_layers}_{current_time}',
                config={
                    'feature_dim': feature_dim,
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
    
    def train(self, enable_plt=False):
        time_train_start = time.time()
        # Optionally, set up a live plot
        if enable_plt:
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
            
            if enable_plt:
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
            if self.enable_wandb:
                self.wandb.log({
                    'Avg Loss': avg_epoch_loss,
                    'time left (min)': time_left/60,
                    'Accuracy': acc_rate,
                }, step=epoch)
        
        if enable_plt:
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
    DATA_TRAIN_LIM = 1
    DATA_VAL_LIM = 1
    ENABLE_WANDB = False


    # Create a DataLoaderFlynet instance
    data_loader_flynet = DataLoaderFlynet(DATA_PARENT_FOLDER, config, printer, interactive=False)
    train_dataset, val_dataset, num_classes, feature_dim = data_loader_flynet.load_all_train_val_test_dataset(data_train_lim=DATA_TRAIN_LIM, data_val_lim=DATA_VAL_LIM)
    
    # Create the Trainer and start training
    trainer = Trainer(config, train_dataset, val_dataset, feature_dim, num_classes, enable_wandb=ENABLE_WANDB)
    trainer.train()


if __name__ == "__main__":
    main()
