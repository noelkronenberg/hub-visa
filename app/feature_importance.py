import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
import numpy as np

from config import RED

def _get_feature_importance(model):
    """
    Calculate feature importance using built-in feature importance of sklearn models.
    """

    # check if the model has feature_importances_ attribute
    importance_values = model.feature_importances_
    logging.info(f"Feature importance values extracted successfully.")

    # if feature_names is not provided, get it from the model
    feature_names = model.feature_names_in_
    logging.info(f"Feature names extracted from the model.")

    # create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': list(feature_names), 
        'importance': importance_values
    })
    logging.info(f"Feature importance DataFrame created successfully.")

    return feature_importance

def visualize_feature_importance(model):
    """
    Visualize feature importance using built-in feature importance of sklearn models.
    """

    # get feature importance data
    feature_importance = _get_feature_importance(model)

    # sort feature importance by absolute value in descending order
    sorted_feature_importance = feature_importance.reindex(feature_importance['importance'].abs().sort_values(ascending=False).index)

    # assign rank based on importance
    sorted_feature_importance['rank'] = range(1, len(sorted_feature_importance) + 1)

    # display feature importance
    with st.expander("**Feature Importance**", expanded=True):

        # allow user to select range of values to show
        num_features = st.slider('Number of Features', min_value=1, max_value=len(sorted_feature_importance), value=20)
        sorted_feature_importance = sorted_feature_importance.head(num_features)
        logging.info(f"Top {num_features} feature importance displayed successfully.")

        # create bar chart
        fig = go.Figure(data=go.Bar(
            x=sorted_feature_importance['rank'],
            y=sorted_feature_importance['importance'],
            marker_color=RED,
            textposition='auto',
            hovertemplate='Rank: %{x}<br>Feature: %{customdata}<br>Importance: %{y:.4f}<extra></extra>',
            customdata=sorted_feature_importance['feature']
        ))

        # update layout
        fig.update_layout(
            title='',
            xaxis_title='Rank',
            yaxis_title='Importance',
            margin=dict(l=20, r=20, t=20, b=20),
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
        logging.info("Feature importance displayed successfully.")