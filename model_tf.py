# Attempt using Tensorflow
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from keras import layers 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image


# Import the pretrained ResNet18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

# TODO uncomment and place further down when ready to test
# model.eval()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train = x_train[:1500]
x_test = x_test[:1500]
y_train = y_train[:1500]
y_test = y_test[:1500]

print(f"X train data type is {x_train[0].shape}")

# Define a custom dataset to handle on-the-fly transformations
class CIFAR100Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])  # Convert to PIL image
        if self.transform:
            img = self.transform(img)  # Apply preprocessing
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# Create dataset and dataloader
train_dataset = CIFAR100Dataset(x_train, y_train, transform=preprocess)
test_dataset = CIFAR100Dataset(x_test, y_test, transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduce batch size if needed
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# train_samples = []
# for img in x_train:
#     img_pil = Image.fromarray(img)
#     img_transform = preprocess(img_pil)
#     train_samples.append(img_transform)

# test_samples = []
# for img in x_test:
#     img_pil = Image.fromarray(img)
#     img_transform = preprocess(img_pil)
#     test_samples.append(img_transform)

# train_input_batch = torch.stack(train_samples)
# test_input_batch = torch.stack(test_samples)

# y_train = torch.tensor(y_train)
# y_test = torch.tensor(y_test)

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
# model.eval()


# move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# for images, labels in train_loader:
#     with torch.no_grad():
#         output = model(images)

# model.fc = nn.Linear(model.fc.in_features, 100)
model.fc = nn.Linear(model.fc.in_features, 100)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # To track the loss

    for inputs, labels in train_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track the running loss
        running_loss += loss.item()

    # Print loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)


# print("loading dataset")
# print(x_train[0])

# # Resize for training of MobileNet
# x_train = tf.image.resize(x_train[0], [224, 224], preserve_aspect_ratio=True)
# # x_test = tf.image.resize(x_test, (224, 224))
# print("resized")

# Normalize pixel values to the range 0-1
# x_train, x_test = x_train / 255.0, x_test / 255.0

# Load CIFAR-100 dataset
# train_sample = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
# test_sample = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

# Normalize pixel values to the range 0-1
# print(x_train[0])

# One-hot encode the labels
# y_train = keras.utils.to_categorical(y_train, 100)
# y_test = keras.utils.to_categorical(y_test, 100)

####################### Show images
# plt.figure() #figsize=(10, 10))
# # for images in train_sample:
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     # plt.imshow(images[0].numpy().astype("uint8"))
#     plt.imshow(np.asarray(x_train[0]))
#     # plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()
########################

# # The DataLoader in PyTorch splits the dataset into batches, each containing 64 samples.
# # DataLoader automatically divides the data into inputs and outputs.
# train_loader = DataLoader(train_sample, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_sample, batch_size=64, shuffle=False)

# # We perform feature extraction instead of full fine-tuning
# # Load pre-trained MobileNetV2 
# weights = MobileNet_V2_Weights.IMAGENET1K_V1
# mobilenet_v2 = models.mobilenet_v2(weights=weights)

# # TODO lookup format of images in CIFAR100 dataset
# plt.figure() #figsize=(10, 10))
# # for images in train_sample:
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     # plt.imshow(images[0].numpy().astype("uint8"))
#     plt.imshow(np.asarray(train_sample[i][0]))
#     # plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()