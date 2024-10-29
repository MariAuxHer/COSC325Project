import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import MobileNet_V2_Weights

# Transformations for CIFAR-100
transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),  # Random cropping with scaling to 128x128
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-100 dataset
train_sample = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
test_sample = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

# The DataLoader in PyTorch splits the dataset into batches, each containing 64 samples.
# DataLoader automatically divides the data into inputs and outputs.
train_loader = DataLoader(train_sample, batch_size=64, shuffle=True)
test_loader = DataLoader(test_sample, batch_size=64, shuffle=False)

# We perform feature extraction instead of full fine-tuning
# Load pre-trained MobileNetV2 
weights = MobileNet_V2_Weights.IMAGENET1K_V1
mobilenet_v2 = models.mobilenet_v2(weights=weights)

# Freeze all layers except the classifier for feature extraction
for parameter in mobilenet_v2.features.parameters():
    parameter.requires_grad = False

# Modify only the final layer to match CIFAR-100
mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features, 100)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilenet_v2 = mobilenet_v2.to(device)

# Train Only the Final Layer
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet_v2.classifier[1].parameters(), lr=0.001)

# Training loop
# Running loss is accumulated per batch within an epoch to get the total loss for the entire dataset during that epoch.
# Then, we divide the accumulated running loss by the number of batches to get the average loss for that epoch.
# The running loss allow us to analyze how well the model is performing over the entire dataset, instead of a single batch.
# It’s useful for tracking model progress and convergence across epochs.

epochs = 10  # Adjust based on results
for epoch in range(epochs):
    mobilenet_v2.train()  # Set model to training mode
    loss = 0.0    # Reset running loss for each epoch
    for inputs, labels in train_loader:  # train_loader for CIFAR-100
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        # Gradients are the derivatives of the loss with respect to the model parameters. We need to zero them at the start of each batch to avoid accumulating gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass
        outputs = mobilenet_v2(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        # Calculates the gradients of the loss function with respect to the model's parameters. 
        loss.backward()

        # Optimizer finds the best solution to update the weights
        optimizer.step()

        # Accumulate the running loss
        loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss/len(train_loader)}")

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
        
        # Obtain the total samples 
        total_samples += labels.size(0)

        # Calculate the number of correct predictions
        correct_samples += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct_samples / total_samples
print(f'Test Accuracy: {accuracy:.2f}%')
