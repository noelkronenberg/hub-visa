import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# load the data
df_target = pd.read_csv('/Users/noelkronenberg/Documents/GitHub/hub-visa/00_EDA/lucas_organic_carbon/target/lucas_organic_carbon_target.csv')
df_training = pd.read_csv('/Users/noelkronenberg/Documents/GitHub/hub-visa/00_EDA/lucas_organic_carbon/training_test/lucas_organic_carbon_training_and_test_data.csv')
df_combined = pd.merge(df_training, df_target, left_index=True, right_index=True)

# Streamlit app
st.title("VisA")

# user input for Random Forest parameters
max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=14)
n_estimators = st.slider('Number of Estimators', min_value=10, max_value=500, value=384)

# user input for percentage of data to use
data_percentage = st.slider('Percentage of Data to Use', min_value=1, max_value=100, value=1)

# spinner while training the model
with st.spinner('Training the model...'):
    # sample the data
    sample_size = int(len(df_combined) * (data_percentage / 100))
    df_sampled = df_combined.sample(n=sample_size, random_state=42)

    # prepare the data
    predictors = df_sampled.columns[:-1]
    target = df_sampled.columns[-1]
    X = df_sampled[predictors]
    y = df_sampled[target]

    # encode categorical target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    # confusion matrix
    unique_labels = label_encoder.classes_
    cm = confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)))

    # display the confusion matrix
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Predicted: {label}' for label in unique_labels],
        y=[f'Actual: {label}' for label in unique_labels],
        colorscale='Viridis',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        xaxis=dict(tickmode='array', tickvals=list(range(len(unique_labels))), ticktext=unique_labels),
        yaxis=dict(tickmode='array', tickvals=list(range(len(unique_labels))), ticktext=unique_labels)
    )

    st.plotly_chart(fig)