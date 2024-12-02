import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('cifar10_perf2.csv')
models = data['Model'].unique()


plt.figure(figsize=(10, 6))

for i, model in enumerate(models):
    model_data = data[data['Model'] == model]
    epochs = model_data['Epoch']

    plt.subplot(2, 1, 1)
    plt.plot(epochs, model_data['Train Loss'], label='Train Loss')
    plt.plot(epochs, model_data['Validation Loss'], label='Validation Loss')
    plt.title(f'{model} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Function')
    plt.ylim(0, 1)
    plt.legend()

for i, model in enumerate(models):
    model_data = data[data['Model'] == model]
    epochs = model_data['Epoch']

    plt.subplot(2, 1, 2)
    plt.plot(epochs, model_data['Train Accuracy'], label='Train Accuracy')
    plt.plot(epochs, model_data['Validation Accuracy'], label='Validation Accuracy')
    plt.title(f'{model} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.ylim(75, 100)
    plt.legend()

plt.tight_layout()
plt.show()