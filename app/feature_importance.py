import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
import numpy as np

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
            marker_color='blue',
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

def _get_interaction_values(model, X_sample, feature_names):
    """
    Calculate interaction values between features using SHAP for binary classification.
    """

    explainer = shap.TreeExplainer(model)
    
    # calculate SHAP interaction values
    shap_interaction_values = explainer.shap_interaction_values(X_sample)
    if len(shap_interaction_values.shape) == 4:
        mean_interactions = np.abs(shap_interaction_values).mean(axis=(0, -1))
    else:
        mean_interactions = np.abs(shap_interaction_values).mean(axis=0)
    
    # create interaction matrix DataFrame
    interaction_matrix = pd.DataFrame(
        mean_interactions,
        columns=feature_names,
        index=feature_names
    )

    # fill diagonal with zeros
    np.fill_diagonal(interaction_matrix.values, 0)
    max_val = np.abs(interaction_matrix.values).max()
    if max_val > 0:
        interaction_matrix = interaction_matrix / max_val
        logging.info(f"Interaction matrix normalized successfully with max value: {max_val}")

    logging.info(f"Feature interaction matrix created successfully with shape: {interaction_matrix.shape}")
    
    return interaction_matrix

def visualize_feature_interactions(model, X_test, feature_names):
    """
    Analyze and visualize feature interactions
    """

    sample_size = min(50, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    interaction_matrix = _get_interaction_values(
        model,
        X_sample,
        feature_names
    )
    
    if interaction_matrix is not None:
        # create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=interaction_matrix.values,
            x=interaction_matrix.columns,
            y=interaction_matrix.index,
            colorscale='RdBu_r',  
            zmid=0,
            showscale=True,
            hoverongaps=False,
            hovertemplate='Feature 1: %{x}<br>Feature 2: %{y}<br>Interaction (SHAP Interaction Value): %{z:.4f}<extra></extra>'
        ))

        # update layout
        fig.update_layout(
            title='',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=800,
            xaxis={'tickangle': 45},
            margin=dict(l=20, r=20, t=20, b=20)
        )

        # display feature interaction
        with st.expander("**Feature Interaction**", expanded=True):
            st.plotly_chart(fig, use_container_width=True)
            logging.info("Feature interaction displayed successfully.")