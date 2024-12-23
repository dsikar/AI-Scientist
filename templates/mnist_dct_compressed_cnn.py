import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from dataclasses import dataclass

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

def train(net, trainloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
                
        print(f'Epoch {epoch+1} completed in {time.time()-start_time:.2f}s')
        
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

@dataclass
class Config:
    # data
    data_path: str = './data'
    # training
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    epochs: int = 10
    # system
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 2
    # logging
    log_interval: int = 100
    # output
    out_dir: str = 'run_0'
    seed: int = 0

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = CompressedDataset(datasets.MNIST(config.data_path, train=True, download=True, transform=transform))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, 
                                            shuffle=True, num_workers=config.num_workers)

    testset = CompressedDataset(datasets.MNIST(config.data_path, train=False, transform=transform))
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, 
                                           shuffle=False, num_workers=config.num_workers)
    
    return trainloader, testloader

def main():
    parser = argparse.ArgumentParser(description="Train DCT Compressed CNN on MNIST")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save/load the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Outputs will be saved to {args.out_dir}")

    # Define number of seeds
    num_seeds = 3  # Run experiment with 3 different seeds for statistical significance
    all_results = {}
    final_infos = {}
    final_info_list = []

    for seed_offset in range(num_seeds):
        config = Config(
            data_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            out_dir=args.out_dir,
            seed=seed_offset
        )

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        print(f"Starting training with seed {seed_offset}")
        start_time = time.time()
        
        trainloader, testloader = get_data_loaders(config)
        net = CompressedNet()
        
        print("Starting training...")
        train(net, trainloader, config.epochs)
        accuracy = test(net, testloader)
        total_time = time.time() - start_time
        
        print(f'Test accuracy for seed {seed_offset}: {accuracy:.2f}%')
        
        # Store results
        final_info = {
            "test_acc": accuracy,
            "total_train_time": total_time,
            "config": vars(config)
        }
        final_info_list.append(final_info)
        
        key_prefix = f"seed_{seed_offset}"
        all_results[f"{key_prefix}_final_info"] = final_info

    # Aggregate results over seeds
    accuracies = [info["test_acc"] for info in final_info_list]
    mean_accuracy = np.mean(accuracies)
    stderr_accuracy = np.std(accuracies) / np.sqrt(len(accuracies))
    
    final_infos = {
        "means": {"test_acc_mean": mean_accuracy},
        "stderrs": {"test_acc_stderr": stderr_accuracy},
        "final_info_dict": {"test_acc": accuracies}
    }

    # Save results
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print(f"All results saved to {args.out_dir}")
    print(f"Mean accuracy across {num_seeds} seeds: {mean_accuracy:.2f}% ± {stderr_accuracy:.2f}%")

if __name__ == "__main__":
    main()
