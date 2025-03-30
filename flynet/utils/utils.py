import wandb
import torch
import os

from python_utils.printer import Printer
global_printer = Printer()

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

def save_model(model, optimizer, model_dir, epoch, losses, this_is_best_model=False):
    if this_is_best_model:
        best_model_dir = os.path.join(model_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        # delete all files in best_model_dir
        for file in os.listdir(best_model_dir):
            file_path = os.path.join(best_model_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(e)
        model_name = f"best_model_e_{epoch}.pth"
        model_save_dir = best_model_dir
    else:
        model_name = f"model_epoch_{epoch+1}.pth"
        model_save_dir = os.path.join(model_dir, f"epoch_{epoch+1}")
        
    os.makedirs(model_save_dir, exist_ok=True)

    model_save_path = os.path.join(model_save_dir, model_name)
    # torch.save(model.state_dict(), model_save_path)
    torch.save({
        'epoch': epoch,
        'model_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'losses': losses,
    }, model_save_path)
    if this_is_best_model:
        if losses:
            print(f"Best model saved with loss: {losses[-1]} to {model_save_path}")
        else:
            global_printer.print_red(f"Best model saved to {model_save_path} but (losses list is empty)")
    else:
        print(f"Model saved at epoch {epoch+1} to {model_save_path}")

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