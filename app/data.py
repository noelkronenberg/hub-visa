import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import logging

# preset data
preset_target = 'app/lucas_organic_carbon/target/lucas_organic_carbon_target.csv'
preset_training = 'app/lucas_organic_carbon/training_test/compressed_data.csv'

# load data
@st.cache_data(show_spinner=False)
def load_data(load_preset_target, load_preset_training, target_data=None, training_data=None):
    
    # load data if not in session state
    if load_preset_target:
        df_target = pd.read_csv(preset_target)
        logging.info("Loaded target data from preset.")
    else:
        df_target = target_data
        logging.info("Loaded target data from uploaded file.")

    if load_preset_training:
        df_training = pd.read_csv(preset_training) # NOTE: this is a compressed version of the data (for demo purposes)
        logging.info("Loaded training data from preset.")
    else:
        df_training = training_data
        logging.info("Loaded training data from uploaded file.")

    df_combined = pd.merge(df_training, df_target, left_index=True, right_index=True)

    return df_training, df_target, df_combined

# prepare data
@st.cache_data(show_spinner=False)
def prepare_data(df_combined, data_percentage):
    # sample the data
    sample_size = int(len(df_combined) * (data_percentage / 100))
    df_sampled = df_combined.sample(n=sample_size, random_state=42)
    predictors = df_sampled.columns[:-1]

    # prepare the data
    target = df_sampled.columns[-1]
    X = df_sampled[predictors]
    y = df_sampled[target]

    # encode categorical target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder