import streamlit as st
import pandas as pd
import os

import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
from pycaret import regression
from pycaret import classification

with st.sidebar:
    st.image("https://www.atriainnovation.com/wp-content/uploads/2021/02/portada.jpg")
    st.title("AutoStream")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Machine-Learning", "Download"])
    st.info("This Application allows you to Upload a Dataset, Perform Machine Learning Training and download the trained model")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your Data for Modelling")
    file = st.file_uploader("Upload your Dataset here!")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Machine-Learning":
    st.title("Perform Machine Learning training on the chosen Dataset")
    new_choice = st.radio("Supervised Learning Method", ["Classification", "Regression"])
    if new_choice == "Regression":
        st.title("Perform Regression")
        target = st.selectbox("Select the target variable", df.columns)
        if st.button("Perform Machine Learning training"):
            regression.setup(df, target=target, fold_shuffle=True, data_split_shuffle=True)
            setup_df = regression.pull()
            st.info("These are the Machine Learning Experiment Settings")
            st.dataframe(setup_df)
            best_model = regression.compare_models()
            compare_df = regression.pull()
            st.info("These are the best trained Models on the Dataset")
            st.dataframe(compare_df)
            best_model
            regression.save_model(best_model, "best_model")
    if new_choice == "Classification":
        st.title("Perform Classification")
        target = st.selectbox("Select the target variable", df.columns)
        if st.button("Perform Machine Learning Training"):
            classification.setup(df, target=target, fold_shuffle=True, data_split_shuffle=True)
            setup_df = classification.pull()
            st.info("These are the Machine Learning Experiment Settings")
            st.dataframe(setup_df)
            best_model = classification.compare_models()
            compare_df = classification.pull()
            st.info("These are the best trained Models on the Dataset")
            st.dataframe(compare_df)
            best_model
            classification.save_model(best_model, "best_model")


if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
