import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from itertools import islice
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
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

####################
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
####################


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
    # print(avg_intensity)

    # pixel_vals.extend(images.cpu().numpy().ravel())
    pixel_vals.extend(avg_intensity)
    if len(pixel_vals) > 1e4: break

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

# count = 0
# counter = 0
# features = []
# tmp = []
# # Principal Component Analysis
# for images, _ in train_loader:
#     for img in images:
#         count += 1
#         tmp = img.cpu().numpy().ravel()
#         counter += len(tmp)
#         print(counter)
#         features.append(tmp)
#         if counter > 1e3: break

# print(f"Num imgs used {count}")
# # features = np.array([img.numpy().ravel() for images, _ in train_loader for img in images])
# pca = PCA(n_components=2)
# features = StandardScaler().fit_transform(features)
# reduced_features = pca.fit_transform(features)

# plt.scatter(reduced_features[:,0], reduced_features[:,1], alpha=0.5)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA of Image Features')
# plt.show()

################

resnet18 = models.resnet18(weights='IMAGENET1K_V1')
with torch.no_grad():
    for images, labels in train_loader:
        features = resnet18(images.to('cpu')).cpu().numpy()
        labels = labels.numpy()

# Working PCA #
count = 0
features = []
targets = []
# Principal Component Analysis
for images, labels in train_loader:
    # labels.append(label)
    for img, label in zip(images, labels):
        count += 1
        features.append(img.cpu().numpy().ravel())
        targets.append(label)
        # tmp = img.cpu().numpy().ravel()
        # counter += len(tmp)
        print(count)
        # features.append(tmp)
        if count > 2000: break
    if count > 2000: break

feat1 = [f.ravel() for f in features]
feat_norm = StandardScaler().fit_transform(feat1)
pca = PCA(n_components=2)
reduced_feats = pca.fit_transform(feat_norm)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    reduced_feats[:, 0],  # PC1
    reduced_feats[:, 1],  # PC2
    c=targets,  # Color by class
    cmap='tab10',  # Colormap for up to 10 classes
    alpha=0.7  # Transparency
)

plt.colorbar(scatter, label='Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Image Features by Class')
plt.show()

# Working LDA #
count = 0
features = []
targets = []

for images, labels in train_loader:
    # labels.append(label)
    for img, label in zip(images, labels):
        count += 1
        features.append(img.cpu().numpy().ravel())
        targets.append(label.item())
        # tmp = img.cpu().numpy().ravel()
        # counter += len(tmp)
        print(count)
        # features.append(tmp)
        if count > 1000: break
    if count > 1000: break

feat1 = [f.ravel() for f in features]
features = np.array(features)
targets = np.array(targets)
scaler = StandardScaler()

feat_norm = scaler.fit_transform(features)
lda = LinearDiscriminantAnalysis(n_components=2)
reduced_feats = lda.fit_transform(feat_norm, targets)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    reduced_feats[:, 0],  # PC1
    reduced_feats[:, 1],  # PC2
    c=targets,  # Color by class
    cmap='Spectral',  # Colormap for up to 10 classes
    alpha=0.7  # Transparency
)

plt.colorbar(scatter, label='Class')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA of Image Features by Class')
plt.show()