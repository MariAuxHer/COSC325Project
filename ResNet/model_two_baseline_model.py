# The second model involved freezing all layers except for the fourth residual block layer, preserving the pre-trained weights in the other layers while allowing adjustments only in the fourth block.
# This is our BASELINE MODEL

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar

# Transformations for CIFAR-100
transform = transforms.Compose([
    # Resize to 224x224 for ResNet
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    # Normalize using ImageNet's mean and std deviation
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  
])

# Load CIFAR-100 dataset
full_dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))  # 80% for training
val_size = len(full_dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
# The DataLoader in PyTorch splits the dataset into batches, each containing 64 samples.
# DataLoader automatically divides the data into inputs and outputs.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet model
resnet18 = models.resnet18(weights='IMAGENET1K_V1')

# Modify the final layer to match CIFAR-100
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 100)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print out the device being used
print(f'Using device: {device}')

resnet18 = resnet18.to(device)

# Unfreeze specific layers (the last block of layers)
for name, param in resnet18.named_parameters():
    # Unfreeze the last residual block
    if 'layer4' in name:  
        param.requires_grad = True

    # Freeze other layers
    else:
        param.requires_grad = False  

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

# Function to train and validate the model
def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_accuracy = []
    validation_accuracy = []
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Set model to training mode
        model.train()  
        epoch_loss = 0
        n_correct = 0
        n_examples = 0

        # Training part of the model
        for examples, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            # Reset gradients
            optimizer.zero_grad()  

            # Forward pass
            logits = model(examples.to(device))  

            # Calculate loss
            loss = criterion(logits, labels.to(device))  

            # Backward pass
            loss.backward()  

            # Update weights
            optimizer.step()  
            
            # Convert logits to predictions
            predicted_labels = torch.argmax(logits, dim=1)
            n_correct += (predicted_labels == labels.to(device)).sum().item()
            epoch_loss += loss.item()
            n_examples += labels.size(0)

        train_accuracy.append(n_correct / n_examples)
        train_loss.append(epoch_loss / len(train_loader))
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy[-1] * 100:.2f}%")

        # Validation part of the model
        model.eval()  # Set model to evaluation mode
        validation_loss = 0
        n_correct = 0
        n_examples = 0

        # Disable gradients for validation
        with torch.no_grad():  
            for examples, labels in tqdm(val_loader, desc="Validating"):

                # Forward pass
                logits = model(examples.to(device))  

                # Calculate loss
                loss = criterion(logits, labels.to(device))  
                
                # Convert logits to predictions
                predicted_labels = torch.argmax(logits, dim=1)
                n_correct += (predicted_labels == labels.to(device)).sum().item()
                validation_loss += loss.item()
                n_examples += labels.size(0)

        validation_accuracy.append(n_correct / n_examples)
        val_loss.append(validation_loss / len(val_loader))
        
        print(f"Validation Loss: {validation_loss / len(val_loader):.4f}, Validation Accuracy: {validation_accuracy[-1] * 100:.2f}%")

# Call the training and validation function
train_and_validate_model(resnet18, train_loader, val_loader, criterion, optimizer, epochs=10)