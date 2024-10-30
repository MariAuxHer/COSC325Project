import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from itertools import islice
from sklearn.decomposition import PCA
from PIL import Image

# Define transformations with resizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 128x128 for better visualization
    transforms.ToTensor()
])

# Load dataset with transformations
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
visual_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

# print(type(train_dataset))
# print(train_dataset[0])     # format: (3d array of pixels, class #)
# print(type(train_loader))
# print(train_loader)

#################################
# Create a DataLoader and visualize some images
images, labels = next(iter(visual_loader))

# Plot images in a grid
img_grid = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10, 10))
plt.imshow(img_grid.permute(1, 2, 0))
plt.title("Resized CIFAR-100 Images")
plt.show()
#################################

# ####################
# Class distribution
all_labels = [label.item() for _, labels in train_loader for label in labels]
num_labels = Counter(all_labels)
# print(num_labels.keys())
# print(num_labels.values())
plt.bar(num_labels.keys(), num_labels.values())
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()
# ####################


# Pixel Intensity Distribution
#   pixel_vals = np.concatenate([img.numpy().ravel() for images, _ in train_loader for img in images])
pixel_vals = np.concatenate([images.cpu().numpy().ravel() for images, _ in train_loader])
plt.hist(pixel_vals, bins=50, range=(0,1), density=True)
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.title('Pixel Intensity Distribution')
plt.show()

################################
pixel_vals = []
count = 0
for images, _ in train_loader:
    count += 1
    # Move images to CPU and flatten them, then append to list
    # Create an avg pixel value for each channel (RGB) then add it to the final array
    avg_intensity = images.mean(dim=(2, 3))

    # pixel_vals.extend(images.cpu().numpy().ravel())
    pixel_vals.extend(avg_intensity)
    if len(pixel_vals) > 1e5: break

print(f"Num imgs used: {count}")
# Convert the list to a NumPy array
pixel_vals = np.array(pixel_vals)

# Plot the pixel intensity distribution
plt.hist(pixel_vals, bins=50, range=(0, 1), density=True)
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.title('Pixel Intensity Distribution')
plt.show()
################################

count = 0
counter = 0
features = []
tmp = []
# Principal Component Analysis
for images, _ in train_loader:
    for img in images:
        count += 1
        tmp = img.cpu().numpy().ravel()
        counter += len(tmp)
        features.append(tmp)
        if counter > 1e5: break

print(f"Num imgs used {count}")
# features = np.array([img.numpy().ravel() for images, _ in train_loader for img in images])
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.scatter(reduced_features[:,0], reduced_features[:,1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Image Features')
plt.show()