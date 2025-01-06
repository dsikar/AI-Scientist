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

class CompressedNet(nn.Module):
    def __init__(self, num_classes=10):  # Added num_classes parameter for framework compatibility
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def compute_gradient_stats(self):
        """Compute gradient L2 norms and correlations"""
        total_norm = 0
        correlation = 0
        
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if hasattr(p, 'prev_grad'):
                    correlation += torch.dot(p.grad.data.view(-1), p.prev_grad.view(-1)).item()
                p.prev_grad = p.grad.data.clone()
                
        return {
            'grad_norm': np.sqrt(total_norm),
            'grad_correlation': correlation
        }
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Removed softmax for compatibility with CrossEntropyLoss

class CompressedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, compression_type='dct'):
        self.dataset = dataset
        self.compression_type = compression_type
        if compression_type == 'random':
            # Initialize random projection matrix (784 -> 256)
            torch.manual_seed(42)  # For reproducibility
            self.projection_matrix = torch.randn(784, 256) / np.sqrt(784)
            
    def __len__(self):
        return len(self.dataset)
        
    def dct_compress(self, image):
        """Apply DCT compression to image"""
        # Reshape and apply DCT
        image = image.view(28, 28)
        dct = self.dct2d(image)
        # Keep top 256 coefficients (16x16)
        mask = torch.zeros_like(dct)
        mask[:16, :16] = 1
        compressed = (dct * mask).flatten()[:256]
        return compressed
        
        
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
        if self.compression_type == 'dct':
            compressed = self.dct_compress(image)
        elif self.compression_type == 'random':
            # Apply random projection
            flattened = image.view(-1)
            compressed = torch.mv(self.projection_matrix.t(), flattened)
        else:
            raise ValueError(f"Unknown compression type: {self.compression_type}")
        return compressed, label

@dataclass
class Config:
    # data
    data_path: str = './data'
    dataset: str = 'mnist'
    num_classes: int = 10
    # model
    model: str = 'compressed_net'
    # training
    batch_size: int = 128
    learning_rate: float = 0.1  # Testing high learning rate for Run 5
    weight_decay: float = 1e-4
    epochs: int = 30  # Increased epochs to better observe training dynamics
    track_gradients: bool = True
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
        compression_type='dct'
    )
    test_dataset = CompressedDataset(
        datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform),
        compression_type='random'
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

    model = CompressedNet(num_classes=config.num_classes).to(config.device)

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
                log_info = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total,
                    'lr': optimizer.param_groups[0]['lr']
                }
                
                if config.track_gradients:
                    grad_stats = model.compute_gradient_stats()
                    log_info.update(grad_stats)
                    
                train_log_info.append(log_info)
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
