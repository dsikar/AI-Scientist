import argparse
import json
import os
import random
import time
import scipy.fftpack
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.training = True  # Add training mode flag
        # Initialize 8 learnable band weights
        self.band_weights = torch.ones(8, requires_grad=True)
        self.optimizer = optim.Adam([self.band_weights], lr=0.001)
        # Define 8 frequency bands based on distance from DC
        self.bands = self._create_frequency_bands(28, 28)
        # Track band weights history
        self.weight_history = []
        self.log_interval = 100  # Log weights every 100 iterations
        self.iterations = 0
        
    def _create_frequency_bands(self, h, w):
        """Create 8 frequency bands based on distance from DC component"""
        y = torch.arange(h).view(-1, 1).float()
        x = torch.arange(w).view(1, -1).float()
        dist = torch.sqrt((y - h//2)**2 + (x - w//2)**2)
        max_dist = torch.max(dist)
        # Create 8 bands of increasing frequency
        bands = []
        for i in range(8):
            lower = max_dist * i / 8
            upper = max_dist * (i + 1) / 8
            band = (dist >= lower) & (dist < upper)
            bands.append(band)
        return bands
        
    def __len__(self):
        return len(self.dataset)
        
    def dct2d(self, x):
        """Compute 2D DCT using scipy and convert back to torch tensor"""
        from scipy.fftpack import dct
        
        # Convert to numpy array
        x_np = x.cpu().numpy()
        
        # Apply 2D DCT
        dct_np = dct(dct(x_np.T, norm='ortho').T, norm='ortho')
        
        # Convert back to torch tensor
        return torch.from_numpy(dct_np).float().to(x.device)
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Reshape and apply DCT
        image = image.view(28, 28)
        dct = self.dct2d(image)
        
        with torch.set_grad_enabled(self.training):
            # Apply softmax to band weights
            band_weights_soft = F.softmax(self.band_weights, dim=0)
            
            # Create weighted mask using frequency bands
            mask = torch.zeros_like(dct)
            for i, band in enumerate(self.bands):
                mask[band] = band_weights_soft[i]
                
            # Apply mask and flatten
            compressed = (dct * mask).flatten()[:256]
            
            # Update band weights based on reconstruction loss
            if self.training:
                self.optimizer.zero_grad()
                recon_loss = F.mse_loss(compressed, dct.flatten()[:256])
                recon_loss.backward()
                self.optimizer.step()
                
                # Log band weights
                self.iterations += 1
                if self.iterations % self.log_interval == 0:
                    self.weight_history.append({
                        'iteration': self.iterations,
                        'weights': band_weights_soft.detach().cpu().numpy().tolist()
                    })
        
        # Detach the compressed tensor before returning
        return compressed.detach(), label

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
        datasets.MNIST(root=config.data_path, train=True, download=True, transform=transform)
    )
    test_dataset = CompressedDataset(
        datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform)
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

        val_loss, val_acc, _ = evaluate(model, test_loader, criterion, config)
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
    
    # Track per-class accuracy
    class_correct = torch.zeros(config.num_classes, device=config.device)
    class_total = torch.zeros(config.num_classes, device=config.device)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            # Update per-class accuracy
            for class_idx in range(config.num_classes):
                mask = targets == class_idx
                class_correct[class_idx] += predicted[mask].eq(targets[mask]).sum().item()
                class_total[class_idx] += mask.sum().item()

    val_loss = val_loss / len(dataloader)
    val_acc = 100. * val_correct / val_total
    
    # Calculate per-class accuracies
    class_accuracies = 100. * class_correct / class_total
    
    return val_loss, val_acc, class_accuracies.cpu().numpy()

def test(config):
    model = CompressedNet(num_classes=config.num_classes).to(config.device)
    if config.compile_model:
        print("Compiling the model for testing...")
        model = torch.compile(model)
    model.load_state_dict(torch.load(os.path.join(config.out_dir, 'best_model.pth')))
    _, test_loader = get_data_loaders(config)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, class_accuracies = evaluate(model, test_loader, criterion, config)
    print(f'Test - Loss: {test_loss:.3f}, Acc: {test_acc:.3f}%')
    return test_loss, test_acc, class_accuracies

def plot_band_weights(weight_history, out_dir):
    """Plot the evolution of band weights during training"""
    if not weight_history:
        return
            
    plt.figure(figsize=(12, 6))
    iterations = [entry['iteration'] for entry in weight_history]
    weights = np.array([entry['weights'] for entry in weight_history])
        
    for i in range(8):
        plt.plot(iterations, weights[:, i], label=f'Band {i}')
            
    plt.title('Frequency Band Weight Evolution During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Band Weight (after softmax)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'band_weights_evolution.png'))
    plt.close()

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

            test_loss, test_acc, class_accuracies = test(config)
            
            # Plot band weights evolution
            train_loader, _ = get_data_loaders(config)
            if hasattr(train_loader.dataset, 'weight_history'):
                plot_band_weights(train_loader.dataset.weight_history, config.out_dir)
            
            # Save per-class accuracies
            np.save(os.path.join(config.out_dir, 'class_accuracies.npy'), class_accuracies)
            
            # Create bar plot of per-class accuracies
            plt.figure(figsize=(10, 6))
            plt.bar(range(10), class_accuracies)
            plt.title('Per-Class Test Accuracy')
            plt.xlabel('Digit Class')
            plt.ylabel('Accuracy (%)')
            plt.xticks(range(10))
            plt.ylim(0, 100)
            for i, acc in enumerate(class_accuracies):
                plt.text(i, acc, f'{acc:.1f}%', ha='center', va='bottom')
            plt.savefig(os.path.join(config.out_dir, 'per_class_accuracy.png'))
            plt.close()

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
