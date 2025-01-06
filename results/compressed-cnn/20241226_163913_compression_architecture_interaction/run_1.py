import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def count_flops(model, input_size=(1, 256)):
    """Rough FLOP counter for our specific architecture"""
    batch_size = 1
    total_flops = 0
    
    # Get model parameters
    conv1_out = model.conv1.out_channels
    conv2_out = model.conv2.out_channels
    
    # Conv1 FLOPs: (in_ch * kernel_size * out_ch * output_size)
    total_flops += input_size[1] * 3 * conv1_out * batch_size
    
    # Conv2 FLOPs
    conv2_input_size = input_size[1] // 2  # After pooling
    total_flops += conv2_input_size * 3 * conv1_out * conv2_out * batch_size
    
    # FC1 FLOPs
    total_flops += (conv2_out * 64) * 128 * batch_size
    
    # FC2 FLOPs
    total_flops += 128 * 10 * batch_size
    
    return total_flops

class CompressedNet(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=0.5):  # Added width_multiplier
        super().__init__()
        base_width = 16
        self.width_multiplier = width_multiplier
        width1 = int(base_width * width_multiplier)
        width2 = int(base_width * 2 * width_multiplier)
        
        self.conv1 = nn.Conv1d(1, width1, 3, padding=1)
        self.conv2 = nn.Conv1d(width1, width2, 3, padding=1)
        # Calculate fc1 input size based on compressed input size and width
        compressed_length = int(28 * 28 * 0.1)  # Using default compression ratio
        feature_length = compressed_length // 4  # After two max pooling layers
        self.fc1 = nn.Linear(width2 * feature_length, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # For activation statistics
        self.activation_stats = {
            'conv1_mean': [], 'conv1_std': [],
            'conv2_mean': [], 'conv2_std': [],
            'fc1_mean': [], 'fc1_std': []
        }
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        
        # Track activation statistics
        conv1_out = F.relu(self.conv1(x))
        self.activation_stats['conv1_mean'].append(conv1_out.mean().item())
        self.activation_stats['conv1_std'].append(conv1_out.std().item())
        
        x = F.max_pool1d(conv1_out, 2)
        
        conv2_out = F.relu(self.conv2(x))
        self.activation_stats['conv2_mean'].append(conv2_out.mean().item())
        self.activation_stats['conv2_std'].append(conv2_out.std().item())
        
        x = F.max_pool1d(conv2_out, 2)
        x = x.view(x.size(0), -1)
        
        fc1_out = F.relu(self.fc1(x))
        self.activation_stats['fc1_mean'].append(fc1_out.mean().item())
        self.activation_stats['fc1_std'].append(fc1_out.std().item())
        
        x = self.fc2(fc1_out)
        return x

class CompressedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, compression_ratio=0.1):
        self.dataset = dataset
        self.compression_ratio = compression_ratio
        self.compressed_size = int(28 * 28 * compression_ratio)
        self.keep_size = int(np.sqrt(self.compressed_size))
        
    def __len__(self):
        return len(self.dataset)
        
    def dct2d(self, x):
        # Implement 2D DCT using FFT
        X1 = torch.fft.fft(x, dim=0)
        X2 = torch.fft.fft(x, dim=1)
        # Create frequency basis
        n1 = x.shape[0]
        n2 = x.shape[1]
        k1 = torch.arange(n1).float()
        k2 = torch.arange(n2).float()
        # Compute DCT weights
        w1 = 2 * torch.exp(-1j * torch.pi * k1 / (2 * n1))
        w2 = 2 * torch.exp(-1j * torch.pi * k2 / (2 * n2))
        # Apply weights
        X1 = X1 * w1.unsqueeze(1)
        X2 = X2 * w2
        # Take real part and normalize
        dct = torch.real(X1 + X2) / (n1 * n2)**0.5
        return dct
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Reshape and apply DCT
        image = image.view(28, 28)
        dct = self.dct2d(image)
        # Keep top coefficients based on compression ratio
        mask = torch.zeros_like(dct)
        mask[:self.keep_size, :self.keep_size] = 1
        compressed = (dct * mask).flatten()[:self.compressed_size]
        return compressed, label

@dataclass
class Config:
    # data
    data_path: str = './data'
    dataset: str = 'mnist'
    num_classes: int = 10
    # model
    model: str = 'compressed_net'
    width_multiplier: float = 0.5
    compression_ratio: float = 0.1
    # training
    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 2
    # system
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 2
    # logging
    log_interval: int = 100
    eval_interval: int = 1000
    # output
    out_dir: str = 'run_0'
    seed: int = 0
    # compile for SPEED!
    compile_model: bool = False

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CompressedDataset(
        datasets.MNIST(root=config.data_path, train=True, download=True, transform=transform),
        compression_ratio=config.compression_ratio
    )
    test_dataset = CompressedDataset(
        datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform),
        compression_ratio=config.compression_ratio
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )

    return train_loader, test_loader

def train(config):
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    model = CompressedNet(
        num_classes=config.num_classes,
        width_multiplier=config.width_multiplier
    ).to(config.device)
    
    # Count and log FLOPs
    flops = count_flops(model, (1, int(28 * 28 * config.compression_ratio)))
    print(f"Model FLOPs: {flops:,}")

    if config.compile_model:
        print("Compiling the model...")
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader, test_loader = get_data_loaders(config)

    best_acc = 0.0
    train_log_info = []
    val_log_info = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % config.log_interval == 0:
                train_log_info.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total,
                    'lr': optimizer.param_groups[0]['lr']
                })
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss / (batch_idx + 1):.3f}, '
                      f'Acc: {100. * train_correct / train_total:.3f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        val_loss, val_acc = evaluate(model, test_loader, criterion, config)
        val_log_info.append({
            'epoch': epoch,
            'loss': val_loss,
            'acc': val_acc
        })
        print(f'Validation - Loss: {val_loss:.3f}, Acc: {val_acc:.3f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'best_model.pth'))

        scheduler.step()

    return train_log_info, val_log_info, best_acc

def evaluate(model, dataloader, criterion, config):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(dataloader)
    val_acc = 100. * val_correct / val_total

    return val_loss, val_acc

def test(config):
    model = CompressedNet(num_classes=config.num_classes).to(config.device)
    if config.compile_model:
        print("Compiling the model for testing...")
        model = torch.compile(model)
    model.load_state_dict(torch.load(os.path.join(config.out_dir, 'best_model.pth')))
    _, test_loader = get_data_loaders(config)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(model, test_loader, criterion, config)
    print(f'Test - Loss: {test_loss:.3f}, Acc: {test_acc:.3f}%')
    return test_loss, test_acc

def main():
    parser = argparse.ArgumentParser(description="Train CompressedNet for Image Classification")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save/load the dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Outputs will be saved to {args.out_dir}")

    # Define datasets and number of seeds per dataset
    datasets = ['mnist']
    num_seeds = {
        'mnist': 1
    }

    all_results = {}
    final_infos = {}

    for dataset in datasets:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            config = Config(
                data_path=args.data_path,
                dataset=dataset,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                out_dir=args.out_dir,
                seed=seed_offset
            )
            os.makedirs(config.out_dir, exist_ok=True)
            print(f"Starting training for {dataset} with seed {seed_offset}")
            start_time = time.time()
            train_log_info, val_log_info, best_acc = train(config)
            total_time = time.time() - start_time

            test_loss, test_acc = test(config)

            final_info = {
                "best_val_acc": best_acc,
                "test_acc": test_acc,
                "total_train_time": total_time,
                "config": vars(config)
            }
            final_info_list.append(final_info)

            key_prefix = f"{dataset}_{seed_offset}"
            all_results[f"{key_prefix}_final_info"] = final_info
            all_results[f"{key_prefix}_train_log_info"] = train_log_info
            all_results[f"{key_prefix}_val_log_info"] = val_log_info

            print(f"Training completed for {dataset} seed {seed_offset}. Best validation accuracy: {best_acc:.2f}%, Test accuracy: {test_acc:.2f}%")

        final_info_dict = {k: [d[k] for d in final_info_list if k in d] for k in final_info_list[0].keys()}
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items() if isinstance(v[0], (int, float))}
        stderrs = {f"{k}_stderr": np.std(v) / np.sqrt(len(v)) for k, v in final_info_dict.items() if isinstance(v[0], (int, float))}
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict
        }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print(f"All results saved to {args.out_dir}")

if __name__ == "__main__":
    main()
