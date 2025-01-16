import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data import preset_training

def visualize_feature_importance(rf_classifier, training_data):
    """
    Display feature importance of a random forest classifier.
    """
    
    # get feature importance
    importances = rf_classifier.feature_importances_
    feature_names = training_data.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    logging.info(f"Feature importance computed successfully.")

    # prepare data for plotting
    sorted_importances = feature_importance_df['Importance'].values
    sorted_features = feature_importance_df['Feature'].values

    # create a horizontal bar chart
    fig = go.Figure(data=go.Bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h'
    ))

    # adjust layout
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=20, b=20),
        height=600
    )

    # display the figure
    with st.expander("**Overall Feature Importance**", expanded=True):
        st.plotly_chart(fig)
        logging.info("Feature importance displayed successfully.")