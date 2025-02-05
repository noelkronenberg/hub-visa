import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import logging

# train model
@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train, max_depth, n_estimators, min_samples_split, min_samples_leaf, max_features):
    """
    Train a Random Forest Classifier model.
    """
    
    rf_classifier = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    
    rf_classifier.fit(X_train, y_train)
    
    return rf_classifier

def evaluate_model(y_test, y_pred, label_encoder=None, normalize_cm=None):
    """
    Evaluate a Random Forest Classifier model.
    """
    
    # compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if label_encoder is not None:
        # actual label names (and save in session state)
        unique_labels = label_encoder.classes_

        # compute confusion matrix (and save in session state)
        if normalize_cm:
            cm = confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)), normalize='all')
        else:
            cm = confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)))

        logging.info("Created confusion matrix.")
    else:
        logging.warning("Label encoder not provided. Unable to compute confusion matrix.")

    logging.info("Model evaluated successfully.")

    if label_encoder is None:
        return accuracy, precision, recall, f1
    else:
        return accuracy, precision, recall, f1, unique_labels, cm
