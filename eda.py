import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms

# Define transformations with resizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 128x128 for better visualization
    transforms.ToTensor()
])

# Load dataset with transformations
cifar100_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader and visualize some images
data_loader = torch.utils.data.DataLoader(cifar100_dataset, batch_size=16, shuffle=True)
images, labels = next(iter(data_loader))

# Plot images in a grid
img_grid = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10, 10))
plt.imshow(img_grid.permute(1, 2, 0))
plt.title("Resized CIFAR-100 Images")
plt.show()
