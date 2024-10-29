import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import MobileNet_V2_Weights

# Transformations for CIFAR-100
transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

# Load CIFAR-100 dataset
train_sample = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
test_sample = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

# The DataLoader in PyTorch splits the dataset into batches, each containing 64 samples.
train_loader = DataLoader(train_sample, batch_size=64, shuffle=True)
test_loader = DataLoader(test_sample, batch_size=64, shuffle=False)

# Load pre-trained MobileNetV2 
weights = MobileNet_V2_Weights.IMAGENET1K_V1
mobilenet_v2 = models.mobilenet_v2(weights=weights)

# Freeze all layers except the classifier
for parameter in mobilenet_v2.features.parameters():
    parameter.requires_grad = False

# Unfreeze the last 5 layers of the feature extractor
for layer in list(mobilenet_v2.features[-5:]):  # Adjust the number to control unfreezing
    for parameter in layer.parameters():
        parameter.requires_grad = True

# Modify only the final layer to match CIFAR-100
mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features, 100)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilenet_v2 = mobilenet_v2.to(device)

# Define the optimizer to update both unfrozen layers and the classifier
params_to_update = [param for param in mobilenet_v2.parameters() if param.requires_grad]
optimizer = optim.Adam(params_to_update, lr=0.0001)  # Use a smaller learning rate

# Define loss function
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10  # Adjust based on results
for epoch in range(epochs):
    mobilenet_v2.train()  # Set model to training mode
    running_loss = 0.0    # Reset running loss for each epoch
    for inputs, labels in train_loader:  # train_loader for CIFAR-100
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = mobilenet_v2(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Testing loop
mobilenet_v2.eval()  # Set model to evaluation mode
correct_samples = 0
total_samples = 0

# No gradient calculation during testing
with torch.no_grad(): 
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = mobilenet_v2(inputs)

        # Get the index of the max log-probability
        _, predicted = torch.max(outputs, 1)  
        
        # Calculate total samples and correct predictions
        total_samples += labels.size(0)
        correct_samples += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct_samples / total_samples
print(f'Test Accuracy: {accuracy:.2f}%')
