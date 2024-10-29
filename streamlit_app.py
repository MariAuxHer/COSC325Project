import streamlit as st

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

elif menu == "About":
    st.title("About This App")
    st.write(
        "Information about app."
    )

