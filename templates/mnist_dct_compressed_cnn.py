import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import os

def save_visualization(original, compressed, epoch, batch_idx, save_dir='visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(121)
    plt.imshow(original.reshape(28, 28).numpy(), cmap='gray')
    plt.title('Original')
    
    # Compressed and reconstructed
    plt.subplot(122)
    # Inverse DCT of compressed
    reconstructed = torch.zeros(28, 28)
    reconstructed[:16, :16] = compressed[:256].reshape(16, 16)
    reconstructed = torch.fft.idctn(reconstructed, norm='ortho')
    plt.imshow(reconstructed.numpy(), cmap='gray')
    plt.title('Reconstructed from DCT')
    
    plt.savefig(f'{save_dir}/comparison_epoch{epoch}_batch{batch_idx}.png')
    plt.close()

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
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Reshape and apply DCT
        image = image.view(28, 28)
        dct = torch.fft.dctn(image, norm='ortho')
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
            if i == 0:  # Save visualization for first batch of each epoch
                original_img = trainloader.dataset.dataset[i][0]  # Get original image
                save_visualization(original_img, inputs[0], epoch, i)
                
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

# Setup data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = CompressedDataset(datasets.MNIST('./data', train=True, download=True, transform=transform))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = CompressedDataset(datasets.MNIST('./data', train=False, transform=transform))
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Train and test
net = CompressedNet()
print("Starting training...")
train(net, trainloader)
accuracy = test(net, testloader)
print(f'Test accuracy: {accuracy:.2f}%')
