import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class SingleChannelDataset(Dataset):
    def __init__(self, dataset, channel=0, mask_type='none', mask_radius=16):
        self.dataset = dataset
        self.channel = channel
        self.mask_type = mask_type
        self.mask_radius = mask_radius
        
        # Pre-compute the mask
        self.mask = None
        if mask_type != 'none':
            self.mask = self._create_mask().to('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_mask(self):
        size = 32  # CIFAR images are 32x32
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
        dist_from_center = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        if self.mask_type == 'center':
            mask = (dist_from_center <= self.mask_radius).float()
        else:  # periphery
            mask = (dist_from_center > self.mask_radius).float()
            
        return mask
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Select single channel from RGB image
        single_channel = image[self.channel].unsqueeze(0)  # Keep channel dim
        
        # Apply mask if needed
        if self.mask_type != 'none':
            device = single_channel.device
            single_channel = single_channel * self.mask.to(device)
            
        return single_channel, label

class SingleChannelCifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 1 x 32 x 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@dataclass
class Config:
    # data
    data_path: str = './data'
    dataset: str = 'cifar10'
    num_classes: int = 10
    channel: int = 0  # 0=R, 1=G, 2=B
    # masking
    mask_type: str = 'none'  # 'none', 'center', 'periphery'
    mask_radius: int = 16  # radius for circular mask (in pixels)
    # model
    model: str = 'single_channel_cnn'
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
    # CIFAR-10 normalization values per channel
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = SingleChannelDataset(
        datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform),
        channel=config.channel
    )
    test_dataset = SingleChannelDataset(
        datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform),
        channel=config.channel
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader

def train(config):
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    model = SingleChannelCifarCNN(num_classes=config.num_classes).to(config.device)

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
    model = SingleChannelCifarCNN(num_classes=config.num_classes).to(config.device)
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
    parser = argparse.ArgumentParser(description="Train Single Channel CNN for CIFAR Classification")
    # parser.add_argument("--data_path", type=str, default="./data", help="Path to save/load the dataset")
    # parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    # parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate")
    # parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    # parser.add_argument("--channel", type=int, default=0, choices=[0, 1, 2], help="Channel to use (0=R, 1=G, 2=B)")
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Outputs will be saved to {out_dir}")

    data_path = './data'
    batch_size = 128
    learning_rate = 0.01
    epochs = 5  # Further reduced to prevent timeout

    # Define experiment configurations
    mask_types = ['center', 'periphery']
    channels = [0, 1, 2]  # R, G, B
    
    all_results = {}
    final_infos = {}
    
    datasets_to_test = ['cifar10']
    for dataset in datasets_to_test:
        final_info_list = []
        for mask_type in mask_types:
            for channel in channels:
                config = Config(
                    data_path=data_path,
                    dataset=dataset,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    out_dir=out_dir,
                    seed=0,
                    mask_type=mask_type,
                    mask_radius=16,
                    channel=channel
                )
                os.makedirs(config.out_dir, exist_ok=True)
                print(f"\n{'='*80}\nStarting training for {dataset} - {mask_type} mask on {['R','G','B'][channel]} channel ({len(mask_types)*channels.index(channel) + mask_types.index(mask_type) + 1}/{len(mask_types)*len(channels)} configurations)\n{'='*80}")
                start_time = time.time()
                train_log_info, val_log_info, best_acc = train(config)
                total_time = time.time() - start_time

                # Run test immediately after training each configuration
                test_loss, test_acc = test(config)
                
                final_info = {
                    "best_val_acc": best_acc,
                    "test_acc": test_acc,
                    "total_train_time": total_time,
                    "mask_type": mask_type,
                    "channel": channel,
                    "config": vars(config)
                }
            final_info_list.append(final_info)

            key_prefix = f"{dataset}_{mask_type}_{['R','G','B'][channel]}"
            all_results[f"{key_prefix}_final_info"] = final_info
            all_results[f"{key_prefix}_train_log_info"] = train_log_info
            all_results[f"{key_prefix}_val_log_info"] = val_log_info

            print(f"Training completed for {dataset} - {mask_type} mask on {['R','G','B'][channel]} channel. Best validation accuracy: {best_acc:.2f}%, Test accuracy: {test_acc:.2f}%")

        final_info_dict = {k: [d[k] for d in final_info_list if k in d] for k in final_info_list[0].keys()}
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items() if isinstance(v[0], (int, float, float))}
        stderrs = {f"{k}_stderr": np.std(v) / np.sqrt(len(v)) for k, v in final_info_dict.items() if isinstance(v[0], (int, float, float))}
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
