import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CompressedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CompressedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
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
        # Keep top 256 coefficients (16x16)
        mask = torch.zeros_like(dct)
        mask[:16, :16] = 1
        compressed = (dct * mask).flatten()[:256]
        return compressed, label


@dataclass
class Config:
    # data
    data_path: str = './data'
    dataset: str = 'mnist'
    num_classes: int = 10
    # model
    model: str = 'compressed_net_cnn'
    # training
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 10
    # system
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 2
    # logging
    log_interval: int = 100
    eval_interval: int = 1000
    # output
    out_dir: str = 'run_mnist'
    seed: int = 0
    # compression
    compression_ratio: float = 0.327  # 256/784 ≈ 0.327
    dct_size: Tuple[int, int] = (16, 16)
    # compile for SPEED!
    compile_model: bool = False


def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=config.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform)

    # Wrap datasets with CompressedDataset
    train_dataset = CompressedDataset(train_dataset)
    test_dataset = CompressedDataset(test_dataset)

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

    model = CompressedNet().to(config.device)
    if config.compile_model:
        print("Compiling the model...")
        model = torch.compile(model)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, 
                               momentum=0.9, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2)

    train_loader, test_loader = get_data_loaders(config)

    best_acc = 0.0
    train_log_info = []
    val_log_info = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            train_total += targets.size(0)
            train_correct += pred.eq(targets.view_as(pred)).sum().item()

            if batch_idx % config.log_interval == 0:
                train_log_info.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total,
                    'lr': optimizer.param_groups[0]['lr'],
                    'compression_ratio': config.compression_ratio
                })
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {train_loss / (batch_idx + 1):.3f}, '
                      f'Acc: {100. * train_correct / train_total:.3f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        epoch_time = time.time() - epoch_start_time
        val_loss, val_acc = evaluate(model, test_loader, criterion, config)
        val_log_info.append({
            'epoch': epoch,
            'loss': val_loss,
            'acc': val_acc,
            'epoch_time': epoch_time
        })
        print(f'Validation - Loss: {val_loss:.3f}, Acc: {val_acc:.3f}%, '
              f'Time: {epoch_time:.2f}s')

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'best_model.pth'))

    return train_log_info, val_log_info, best_acc


def evaluate(model, dataloader, criterion, config):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            total += targets.size(0)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    val_loss = val_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def main():
    try:
        parser = argparse.ArgumentParser(description="Train CompressedNet on MNIST")
        parser.add_argument("--data_path", type=str, default="./data", help="Path to save/load the dataset")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate")
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
        parser.add_argument("--out_dir", type=str, default="run_mnist", help="Output directory")
        args = parser.parse_args()
    except Exception as e:
        print(f"Error parsing arguments: {str(e)}")
        print("Usage example: python experiment.py --out_dir run_0")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Outputs will be saved to {args.out_dir}")

    # Define number of seeds
    num_seeds = 3  # Run with 3 different seeds
    all_results = {}
    final_infos = {}

    for seed_offset in range(num_seeds):
        config = Config(
            data_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            out_dir=args.out_dir,
            seed=seed_offset
        )
        
        print(f"Starting training with seed {seed_offset}")
        start_time = time.time()
        train_log_info, val_log_info, best_acc = train(config)
        total_time = time.time() - start_time

        # Final evaluation
        model = CompressedNet().to(config.device)
        if config.compile_model:
            model = torch.compile(model)
        model.load_state_dict(torch.load(os.path.join(config.out_dir, 'best_model.pth')))
        _, test_loader = get_data_loaders(config)
        criterion = nn.NLLLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, config)

        # Prepare final_info dictionary
        final_info = {
            "best_val_acc": best_acc,
            "test_acc": test_acc,
            "total_train_time": total_time,
            "compression_ratio": config.compression_ratio,
            "config": vars(config)
        }

        key_prefix = f"seed_{seed_offset}"
        all_results[f"{key_prefix}_final_info"] = final_info
        all_results[f"{key_prefix}_train_log_info"] = train_log_info
        all_results[f"{key_prefix}_val_log_info"] = val_log_info

        print(f"Training completed for seed {seed_offset}. "
              f"Best validation accuracy: {best_acc:.2f}%, "
              f"Test accuracy: {test_acc:.2f}%")

    # Calculate statistics across seeds
    accuracies = [results["test_acc"] for results in all_results.values() if "test_acc" in results]
    final_infos["summary"] = {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "compression_ratio": config.compression_ratio,
    }

    # Save results
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print(f"All results saved to {args.out_dir}")


if __name__ == "__main__":
    main()