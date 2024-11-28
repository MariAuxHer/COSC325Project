import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('modelperformance.csv')
models = data['Model'].unique()


plt.figure(figsize=(15, 10))

for i, model in enumerate(models):
    model_data = data[data['Model'] == model]
    epochs = model_data['Epoch']


    plt.subplot(len(models), 1, i + 1)
    plt.plot(epochs, model_data['Train Loss'], label='Train Loss')
    plt.plot(epochs, model_data['Validation Loss'], label='Validation Loss')
    plt.title(f'{model} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.ylim(35, 100)
    plt.legend()

plt.tight_layout()
plt.show()