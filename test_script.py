import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# assert torch.cuda.is_available(), "Use GPU!"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device : ", device)

# Define a simple neural network with various activation functions
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # ReLU activation
        
        # Hidden layers with different activations
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
            # Different activation functions for different layers
            if i % 3 == 0:
                layers.append(nn.ReLU())  # ReLU activation
            elif i % 3 == 1:
                layers.append(nn.Tanh())  # Tanh activation
            else:
                layers.append(nn.LeakyReLU(0.1))  # Leaky ReLU with 0.1 negative slope
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Softmax(dim=1))  # Softmax for classification
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Load MNIST dataset as an example
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Create model, define loss function and optimizer
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_sizes = [512, 256, 128, 64]
output_size = 10  # 10 digits

model = SimpleNN(input_size, hidden_sizes, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Reshape data: [batch_size, 1, 28, 28] -> [batch_size, 784]
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            output = model(data)
            target = target.to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Test after each epoch
        test()

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            output = model(data)
            target = target.to(device)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# Run 3 epochs of training
if __name__ == "__main__":
    print("Starting training...")
    train(3)