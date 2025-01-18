import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import numpy as np

from data import preset_training


def get_feature_importance(model, X_test, feature_names):
    """Calculate feature importance using SHAP values"""
    try:
        explainer = shap.TreeExplainer(model)
   
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        # Get SHAP values
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
        
        # Ensure importance_values length matches feature_names length
        if len(importance_values) != len(feature_names):
            logging.error(f"Mismatch: importance_values length ({len(importance_values)}) != feature_names length ({len(feature_names)})")
            if len(importance_values) > len(feature_names):
                importance_values = importance_values[:len(feature_names)]
            else:
                importance_values = np.pad(importance_values, 
                                         (0, len(feature_names) - len(importance_values)), 
                                         'constant')

        feature_importance = pd.DataFrame({
            'feature': list(feature_names), 
            'importance': importance_values
        })
        
        
        logging.info(f"Feature importance DataFrame created successfully with shape: {feature_importance.shape}")
        return feature_importance, explainer, X_sample
        
    except Exception as e:
        logging.error(f"Error in feature importance calculation: {str(e)}")
        logging.error(f"Importance values shape: {importance_values.shape if 'importance_values' in locals() else 'not calculated'}")
        logging.error(f"Feature names length: {len(feature_names)}")
        raise e

def visualize_feature_importance(model, X_test, feature_names):
    """Visualize feature importance using SHAP values"""
    # Get feature importance data
    feature_importance, explainer, X_sample = get_feature_importance(model, X_test, feature_names)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Create bar chart
        fig = go.Figure(data=go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker_color=['red' if x < 0 else 'blue' for x in feature_importance['importance']],
            text=feature_importance['importance'].round(4),
            textposition='auto',
        ))

        fig.update_layout(
            title='SHAP Feature Importance ',
            xaxis_title='Importance (SHAP Value)',
            yaxis_title="Feature",
            margin=dict(l=20, r=20, t=20, b=20),
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("Detailed SHAP Feature Importance Scores:")
        formatted_df = feature_importance.copy()
        formatted_df = formatted_df[formatted_df['importance'].abs() > 0.0001]
        formatted_df = formatted_df.reindex(
            formatted_df['importance'].abs().sort_values(ascending=False).index
        )
        formatted_df['importance'] = formatted_df['importance'].round(4)
        
        # Create DataFrame with specific columns and reset index
        final_df = pd.DataFrame({
            'Rank': range(1, len(formatted_df) + 1),
            'Feature': formatted_df['feature'],
            'Importance (SHAP Value)': formatted_df['importance']
        }).reset_index(drop=True) 
        
        st.dataframe(
            final_df,
            height=600,
            use_container_width=True,
            hide_index=True  
        )
        
        st.write("""
        **SHAP Value :**
        - SHAP values show each feature's contribution to predictions
        - Positive values indicate the feature increases the prediction
        - Negative values indicate the feature decreases the prediction
        - Larger absolute values indicate stronger impact
        """)
    
    logging.info("SHAP feature importance analysis displayed successfully")

def get_interaction_values(model, X_sample, feature_names):
    """Calculate interaction values between features using SHAP for binary classification"""
    explainer = shap.TreeExplainer(model)
    
    try:
        # Calculate SHAP interaction values
        shap_interaction_values = explainer.shap_interaction_values(X_sample)
        if len(shap_interaction_values.shape) == 4:
            mean_interactions = np.abs(shap_interaction_values).mean(axis=(0, -1))
        else:
            mean_interactions = np.abs(shap_interaction_values).mean(axis=0)
        
        # Create interaction matrix DataFrame
        interaction_matrix = pd.DataFrame(
            mean_interactions,
            columns=feature_names,
            index=feature_names
        )
    
        np.fill_diagonal(interaction_matrix.values, 0)
        max_val = np.abs(interaction_matrix.values).max()
        if max_val > 0:
            interaction_matrix = interaction_matrix / max_val
        
        return interaction_matrix
        
    except Exception as e:
        logging.error(f"Error calculating interaction values: {str(e)}")
        logging.error(f"Input data shape: {X_sample.shape}")
        logging.error(f"SHAP interaction values shape: {shap_interaction_values.shape}")
        return None

def visualize_feature_interactions(model, X_test, feature_names):
    """Analyze and visualize feature interactions"""
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sample_size = min(50, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)

            interaction_matrix = get_interaction_values(
                model,
                X_sample,
                feature_names
            )
            
            if interaction_matrix is not None:
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=interaction_matrix.values,
                    x=interaction_matrix.columns,
                    y=interaction_matrix.index,
                    colorscale='RdBu_r',  
                    zmid=0,
                    showscale=True,
                    text=np.round(interaction_matrix.values, 4),
                    texttemplate='%{text}',
                    textfont={"size": 8},
                    hoverongaps=False
                ))

                fig.update_layout(
                    title='Feature Interaction Strength Heatmap',
                    xaxis_title='Features',
                    yaxis_title='Features',
                    width=800,
                    height=800,
                    xaxis={'tickangle': 45},
                    margin=dict(t=50, l=50, r=50, b=100)  
                )

                st.plotly_chart(fig)
                
                # Find strongest interaction
                interaction_values = interaction_matrix.values.copy()
                max_interaction_idx = np.unravel_index(
                    np.argmax(np.abs(interaction_values)),
                    interaction_values.shape
                )
                feature1 = interaction_matrix.index[max_interaction_idx[0]]
                feature2 = interaction_matrix.columns[max_interaction_idx[1]]
                max_interaction_value = interaction_values[max_interaction_idx]
                
                st.write(f"Strongest Feature Interaction: {feature1} and {feature2} (Strength: {max_interaction_value:.4f})")
        
        with col2:
            st.write("""
            **Feature Interaction Analysis Guide:**
            
            **Heatmap Interpretation:**
            - Darker colors indicate stronger interactions
            - Red indicates positive interaction (synergistic effect)
            - Blue indicates negative interaction (antagonistic effect)
            - Diagonal shows feature self-interaction
            
            **Interaction Meanings:**
            - Strong: Feature combinations significantly impact predictions
            - Positive: Features enhance each other's effects
            - Negative: Features counteract each other's effects
            - Weak: Features act independently
            """)
            
            # Display interaction matrix values
            st.write("Detailed Interaction Strengths:")
            if interaction_matrix is not None:
                st.dataframe(
                    interaction_matrix.style.format("{:.4f}"),
                    height=400
                )

    except Exception as e:
        logging.error(f"Error in feature interaction analysis: {str(e)}")
        st.error("Error occurred during feature interaction analysis. Please check data and model.")

 


