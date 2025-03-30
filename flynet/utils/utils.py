import wandb
import torch

def init_wandb(project_name, run_name, config, run_id=None, resume=None, wdb_notes=''):        
    wandb.init(
        # set the wandb project where this run will be logged
        project = project_name,
        name=run_name,
        # id='o5qeq1n8', resume='allow',
        id=run_id, resume=resume,
        # track hyperparameters and run metadata
        config=config
    )
    # wandb_run_url = wandb.run.url
    return wandb

# Create PyTorch dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, indices=None, file_paths=None):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)  # Move to device (GPU or CPU)
        self.y = torch.tensor(y, dtype=torch.long).to(device)  # Move to device (GPU or CPU)
        self.indices = indices  # List of start indices
        self.file_paths = file_paths  # List of file paths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx], self.file_paths[idx]  # Return index and file path as well