import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import numpy as np

from data import preset_training

def _get_feature_importance(model, X_test, feature_names):
    """
    Calculate feature importance using SHAP values.
    """

    explainer = shap.TreeExplainer(model)

    sample_size = min(100, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    # get SHAP values
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        importance_values = []
        for sv in shap_values:
            if len(sv.shape) > 2:
                sv = sv.reshape(sv.shape[0], -1)
            importance_values.append(sv.mean(axis=0))
        importance_values = np.mean(importance_values, axis=0)
    else:
        if len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
        importance_values = shap_values.mean(axis=0)
    
    # ensure importance_values length matches feature_names length
    if len(importance_values) != len(feature_names):
        logging.error(f"Mismatch: importance_values length ({len(importance_values)}) != feature_names length ({len(feature_names)})")
        if len(importance_values) > len(feature_names):
            importance_values = importance_values[:len(feature_names)]
            logging.warning(f"Importance values truncated to match feature names length.")
        else:
            importance_values = np.pad(importance_values, 
                                        (0, len(feature_names) - len(importance_values)), 
                                        'constant')
            logging.warning(f"Importance values padded to match feature names length.")

    # create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': list(feature_names), 
        'importance': importance_values
    })
    logging.info(f"Feature importance DataFrame created successfully.")

    return feature_importance, explainer, X_sample

def visualize_feature_importance(model, X_test, feature_names, n_features=25):
    """
    Visualize feature importance using SHAP values.
    """

    # get feature importance data
    feature_importance, explainer, X_sample = _get_feature_importance(model, X_test, feature_names)

    # sort feature importance by absolute value in descending order
    sorted_feature_importance = feature_importance.copy()
    sorted_feature_importance = sorted_feature_importance[sorted_feature_importance['importance'].abs() > 0.0001]
    sorted_feature_importance = sorted_feature_importance.reindex(
        sorted_feature_importance['importance'].abs().sort_values(ascending=False).index
    )
    sorted_feature_importance['importance'] = sorted_feature_importance['importance'].round(4)
    logging.info(f"Feature importance DataFrame sorted.")
    
    # create DataFrame with specific columns and reset index
    ranked_feature_importance = pd.DataFrame({
        'rank': range(1, len(sorted_feature_importance) + 1),
        'feature': sorted_feature_importance['feature'],
        'importance': sorted_feature_importance['importance']
    }).reset_index(drop=True).head(n_features)
    logging.info(f"Ranked feature importance DataFrame created.")

    # create bar chart
    fig = go.Figure(data=go.Bar(
        x=ranked_feature_importance['importance'],
        y=ranked_feature_importance['rank'],
        orientation='h',
        marker_color=['red' if x < 0 else 'blue' for x in ranked_feature_importance['importance']],
        text=ranked_feature_importance['importance'].round(4),
        textposition='auto',
        hovertemplate='Feature: %{y}<br>Importance (SHAP Value): %{x:.4f}<extra></extra>'
    ))

    # update layout
    fig.update_layout(
        title='',
        xaxis_title='Importance (SHAP Value)',
        yaxis_title="Rank",
        yaxis=dict(autorange='reversed'), # ensure ranking is from top to bottom
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        showlegend=False
    )

    # display feature importance
    with st.expander("**Feature Importance**", expanded=True):
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
            st.plotly_chart(fig)
            logging.info("Feature interaction displayed successfully.")