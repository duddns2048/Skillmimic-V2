from isaacgym.torch_utils import *
from skillmimic.utils import torch_utils

import os
import numpy as np
import argparse
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader, random_split
import wandb


def compute_humanoid_observations(root_pos, root_rot, body_pos):
    root_h_obs = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    obs = torch.cat((root_h_obs, local_body_pos), dim=-1)
    return obs

def compute_obj_observations(root_pos, root_rot, tar_pos):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)

    return local_tar_pos

class HistoryEncoder(nn.Module):

    def __init__(self, history_length, input_size, embedding_dim):
        super(HistoryEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * history_length, embedding_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, sequence_length)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ComprehensiveModel(pl.LightningModule):

    def __init__(self, history_length, input_size=394, embedding_dim=3, lr=0.001):
        super(ComprehensiveModel, self).__init__()
        self.save_hyperparameters()
        self.history_encoder = HistoryEncoder(history_length, input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + input_size + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 394)

    def forward(self, history, current_motion, current_label):
        history_features = self.history_encoder(history)
        x = torch.cat((history_features, current_motion, current_label), dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        history, current_motion, current_label, y = batch
        y_hat = self(history, current_motion, current_label)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        history, current_motion, current_label, y = batch
        y_hat = self(history, current_motion, current_label)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
            'monitor': 'train_loss',  # Monitor validation loss for scheduling
        }
        return [optimizer], [scheduler]


class CustomDataset(Dataset):
    def __init__(self, motion_dir, history_length=30):
        self.history_length = history_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_dof = 52
        self.file_paths = [motion_dir] if os.path.isfile(motion_dir) else [ \
            os.path.join(root, f) 
            for root, dirs, filenames in os.walk(motion_dir) 
            for f in filenames 
            if f.endswith('.pt')
        ]
        
        print(f'############################ Have load {len(self.file_paths)} motions ############################')
        self.data = []
        
        for file_path in self.file_paths:
            source_data = torch.load(file_path)  # (seq_len, 337)
            source_state = self.data_to_state(source_data) # (seq_len, 808)
            nframe, dim = source_state.shape

            current_motion_data = source_state[:-1]
            target_data = source_state[1:]
            
            skill_number = int(os.path.basename(file_path).split('_')[0].strip('pickle'))
            current_label_data = torch.nn.functional.one_hot(torch.tensor(skill_number), num_classes=64)
            
            history_data = torch.zeros(nframe, history_length, dim)
            for i in range(current_motion_data.shape[0]):
                if i < history_length:
                    history_data[i, history_length-i:] = source_state[:i]
                else:
                    history_data[i] = source_state[i-history_length:i]
                self.data.append((history_data[i], current_motion_data[i], current_label_data, target_data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def data_to_state(self, data):

        nframes = data.shape[0]
        root_pos = data[:, :3]
        root_rot = data[:, 3:6]
        body_pos = data[:, 189:189+71*3].reshape(nframes, 71, 3)
        humanoid_obs = compute_humanoid_observations(root_pos, root_rot, body_pos) # (nframes, 211)
        humanoid_obs = torch.cat((humanoid_obs, data[:, 9:9+180]), dim=-1) # (nframes, 391)
        obj_pos = data[:, 402:405] # (nframes, 3)
        obj_obs = compute_obj_observations(root_pos, root_rot, obj_pos) # (nframes, 3)
        state = torch.cat((humanoid_obs, obj_obs), dim=-1) # (nframes, 394)

        return state

class MotionDataModule(pl.LightningDataModule):
    def __init__(self, folder_path, window_size, batch_size=32):
        super().__init__()
        self.folder_path = folder_path
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Initialize dataset and split it here
        dataset = CustomDataset(self.folder_path, self.window_size)
        
        # Randomly split into training and validation sets
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class MotionDataModuleAll4Train(pl.LightningDataModule):
    def __init__(self, folder_path, window_size, batch_size=32):
        super().__init__()
        self.folder_path = folder_path
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Initialize dataset here
        self.dataset = CustomDataset(self.folder_path, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train state prediction model for ParaHome')
    parser.add_argument('--motion_dir', type=str, default='skillmimic/data/motions/ParaHome/',
                      help='Directory containing motion data')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training')
    parser.add_argument('--history_length', type=int, default=60,
                      help='Length of motion history')
    parser.add_argument('--embedding_dim', type=int, default=3,
                      help='Dimension of motion embedding')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=3000,
                      help='Maximum number of training epochs')
    parser.add_argument('--val_split', action='store_true',
                      help='Use validation split (default: use all data for training)')
    parser.add_argument('--output_dir', type=str, default='hist_encoder/ParaHome',
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--wandb_project', type=str, default='sm2_parahome_hist_encoder',
                      help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='W&B run name (default: History{HL}-Embedding{ED})')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Choose data module based on validation split flag
    if args.val_split:
        data_module = MotionDataModule(args.motion_dir, args.history_length, args.batch_size)
    else:
        data_module = MotionDataModuleAll4Train(args.motion_dir, args.history_length, args.batch_size)

    # Model training
    model = ComprehensiveModel(
        history_length=args.history_length,
        input_size=394,
        embedding_dim=args.embedding_dim,
        lr=args.lr
    )

    name = f"History{args.history_length}-Embedding{args.embedding_dim}"
    run_name = args.wandb_name if args.wandb_name else name

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.output_dir}/checkpoints',
        filename='{epoch}-{train_loss:.6f}',
        monitor='train_loss',
        save_top_k=2,
        mode='min'
    )

    # W&B Logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=args.output_dir,
        config={
            'motion_dir': args.motion_dir,
            'batch_size': args.batch_size,
            'history_length': args.history_length,
            'embedding_dim': args.embedding_dim,
            'lr': args.lr,
            'max_epochs': args.max_epochs,
            'input_size': 394,
            'val_split': args.val_split
        }
    )

    # Create Trainer and train model
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        devices=1,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, data_module)

### Command ###
# Default usage (uses all default values):
# CUDA_VISIBLE_DEVICES=1 python skillmimic/utils/state_prediction_parahome.py

# Custom usage with W&B configuration:
# CUDA_VISIBLE_DEVICES=1 python skillmimic/utils/state_prediction_parahome.py \
#     --motion_dir skillmimic/data/motions/ParaHome/ \
#     --batch_size 256 \
#     --history_length 60 \
#     --embedding_dim 3 \
#     --lr 0.001 \
#     --max_epochs 3000 \
#     --output_dir hist_encoder/ParaHome \
#     --wandb_project parahome-state-prediction \
#     --wandb_name my-experiment-name

# Add --val_split flag to use 85/15 train/val split
# Note: Make sure to login to W&B first: wandb login
###############
