import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


st.sidebar.title("Menu")
menu = st.sidebar.radio("Select a page:", ["Home", "Results", "About"])

if menu == "Home":
    st.title("Welcome to the Decypher App Showcase")
    st.write(
        "This website is to showcase the results our group has made on our Final Project."
)

if menu == "Results":
    st.title("Results")
    st.write(
        "This page is dedicated to showcasing our results and charts."
)

    data = pd.read_csv('modelperformance.csv')
    models = data['Model'].unique()

    
    plt.figure(figsize=(15, 10))

    for i, model in enumerate(models):
        model_data = data[data['Model'] == model]
        epochs = model_data['Epoch']

        plt.subplot(len(models), 1, i + 1)
        plt.plot(epochs, model_data['Train Accuracy'], label='Train Accuracy')
        plt.plot(epochs, model_data['Validation Accuracy'], label='Validation Accuracy')
        plt.title(f'{model} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(35, 100)
        plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())


elif menu == "About":
    st.title("About This App")
    st.write(
        "Information about app."
    )

