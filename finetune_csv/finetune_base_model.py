import os
import sys
import json
import time
import pickle
import random
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from logging.handlers import RotatingFileHandler
import logging
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Kronos, KronosTokenizer
from config_loader import CustomFinetuneConfig

# --- SUPER CHARGED DATASET CLASS ---
class CustomKlineDataset(Dataset):
    def __init__(self, data_path, data_type='train', lookback_window=90, predict_window=10, 
                 clip=5.0, seed=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_path = data_path
        self.data_type = data_type
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        self.py_rng = random.Random(seed)

        self.all_data_segments = [] 
        self.sample_indices = []    
        
        # 1. Load Files (Directory or Single File)
        if os.path.isdir(self.data_path):
            files = sorted(glob.glob(os.path.join(self.data_path, "*.csv")))
        else:
            files = [self.data_path]

        print(f"[{data_type.upper()}] Loading {len(files)} files/timeframes...")
        temp_file_lengths = [] 

        for file_p in files:
            try:
                df = pd.read_csv(file_p)
                df.columns = [c.lower() for c in df.columns] # Lowercase cols
                
                # Auto-detect timestamp column
                ts_col = next((c for c in df.columns if c in ['timestamp', 'timestamps', 'opentime', 'date', 'datetime']), None)
                if ts_col: df.rename(columns={ts_col: 'timestamps'}, inplace=True)
                
                df['timestamps'] = pd.to_datetime(df['timestamps'])
                df = df.sort_values('timestamps').reset_index(drop=True)
                
                # Feature Engineering
                df['minute'] = df['timestamps'].dt.minute
                df['hour'] = df['timestamps'].dt.hour
                df['weekday'] = df['timestamps'].dt.weekday
                df['day'] = df['timestamps'].dt.day
                df['month'] = df['timestamps'].dt.month
                
                if 'volume' not in df.columns: df['volume'] = 0.0
                if 'amount' not in df.columns: df['amount'] = df['close'] * df['volume']

                # Split
                total_len = len(df)
                train_end = int(total_len * train_ratio)
                val_end = int(total_len * (train_ratio + val_ratio))

                if data_type == 'train': target_df = df.iloc[:train_end]
                elif data_type == 'val': target_df = df.iloc[train_end:val_end]
                else: target_df = df.iloc[val_end:]

                if len(target_df) > self.window:
                    self.all_data_segments.append(target_df)
                    valid_samples = len(target_df) - self.window + 1
                    temp_file_lengths.append(valid_samples)
                else:
                    temp_file_lengths.append(0)
            except Exception as e:
                print(f"Skipping file {file_p} due to error: {e}")

        # 2. Balanced Sampling Logic
        if data_type == 'train' and len(temp_file_lengths) > 0:
            max_len = max(temp_file_lengths)
            for segment_id, length in enumerate(temp_file_lengths):
                if length == 0: continue
                # Dynamic Oversampling for MTF Balance
                repeat_factor = int(max_len / length) 
                repeat_factor = min(repeat_factor, 100) 
                repeat_factor = max(repeat_factor, 1)

                indices = [(segment_id, i) for i in range(length)]
                self.sample_indices.extend(indices * repeat_factor)
            self.py_rng.shuffle(self.sample_indices)
        else:
            for segment_id, length in enumerate(temp_file_lengths):
                indices = [(segment_id, i) for i in range(length)]
                self.sample_indices.extend(indices)

        self.n_samples = len(self.sample_indices)
        print(f"[{data_type.upper()}] Final Balanced Dataset Size: {self.n_samples}")

    def set_epoch_seed(self, epoch):
        self.py_rng.seed(self.seed + epoch)
        if self.data_type == 'train':
            self.py_rng.shuffle(self.sample_indices)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        segment_id, start_idx = self.sample_indices[idx]
        segment_data = self.all_data_segments[segment_id]
        window_data = segment_data.iloc[start_idx : start_idx + self.window]
        
        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)
        
        # Norm & Clip
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)
        
        return torch.from_numpy(x), torch.from_numpy(x_stamp)

# --- STANDARD LOGGING & UTILS ---
def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    
    log_file = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if console_handler: console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console_handler: logger.addHandler(console_handler)
    return logger

def create_dataloaders(config):
    train_dataset = CustomKlineDataset(config.data_path, 'train', config.lookback_window, config.predict_window, 
                                     config.clip, config.seed, config.train_ratio, config.val_ratio, config.test_ratio)
    val_dataset = CustomKlineDataset(config.data_path, 'val', config.lookback_window, config.predict_window, 
                                   config.clip, config.seed + 1, config.train_ratio, config.val_ratio, config.test_ratio)
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None), 
                            num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                          num_workers=config.num_workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler

# --- TRAINING LOOP ---

def train_model(model, tokenizer, device, config, save_dir, logger):
    use_ddp = dist.is_available() and dist.is_initialized()
    train_loader, val_loader, train_dataset, val_dataset, train_sampler = create_dataloaders(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.predictor_learning_rate, 
                                 betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.adam_weight_decay)
    
    # OneCycleLR needs to know total steps. If accumulating, steps are reduced.
    accum_steps = getattr(config, 'accumulation_steps', 1)
    total_steps = int(len(train_loader) * config.basemodel_epochs / accum_steps)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.predictor_learning_rate, 
                                                   total_steps=total_steps, 
                                                   pct_start=0.03, div_factor=10)
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float('inf')
    
    for epoch in range(config.basemodel_epochs):
        model.train()
        train_dataset.set_epoch_seed(epoch * 10000)
        if train_sampler: train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        optimizer.zero_grad() # Reset gradients at start of epoch
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x, batch_x_stamp = batch_x.to(device, non_blocking=True), batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            
            # --- UNIVERSAL GRADIENT ACCUMULATION LOGIC ---
            # 1. Scale loss
            loss = loss / accum_steps 
            loss.backward()
            
            # 2. Step only after N batches (or at end of epoch)
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), 3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Restore loss for logging purposes
            epoch_loss += loss.item() * accum_steps

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x, batch_x_stamp = batch_x.to(device, non_blocking=True), batch_x_stamp.to(device, non_blocking=True)
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                v_loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                val_loss += v_loss.item()
                val_batches += 1
        
        avg_val = val_loss / val_batches if val_batches > 0 else 0
        if use_ddp:
             dist.all_reduce(torch.tensor(avg_val).to(device))
             avg_val = avg_val / dist.get_world_size()

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1} | Val Loss: {avg_val:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_path = os.path.join(save_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                # Ensure .safetensors saving
                (model.module if use_ddp else model).save_pretrained(save_path, safe_serialization=True)
                print(f"Saved Best Model to {save_path}")

    return best_val_loss

# --- MAIN ---
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = CustomFinetuneConfig(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.basemodel_save_path, exist_ok=True)
    logger = setup_logging(config.exp_name, os.path.join(config.base_save_path, "logs"), 0)
    
    # Load Models
    tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path).to(device)
    model = Kronos.from_pretrained(config.pretrained_predictor_path).to(device)
    
    print("Starting A40 Optimized Training...")
    train_model(model, tokenizer, device, config, config.basemodel_save_path, logger)

if __name__ == "__main__":
    main()