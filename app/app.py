import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import logging
import pickle

from error_analysis import visualize_error_analysis
from feature_importance import visualize_feature_importance

from data import preset_target, preset_training, load_data, prepare_data, plot_target_distribution, plot_spectral_profiles, plot_wavelength_distribution
from model import train_model, evaluate_model

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# streamlit app
st.set_page_config(page_title="VisA Dashboard", layout="wide")
st.title("VisA Dashboard")

st.write("""
    This application allows you to upload your own data, train a machine learning model, 
    and explore the results through various visualizations. Start by (optionally) uploading your data 
    and configuring the model parameters in the sidebar. When the desired results are achieved, 
    you can download the trained model.
""")

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------

st.sidebar.header("Settings")

# sidebar for data exploration
with st.sidebar.expander("**Data**", expanded=False):
    # file upload
    target_file = st.file_uploader("Upload Target CSV", type=["csv"])
    training_file = st.file_uploader("Upload Training CSV", type=["csv"])

    # load data
    if target_file is not None:
        df_target = pd.read_csv(target_file)
        st.session_state['target_data'] = df_target
        logging.info("Added target file to session state.")

    # load data
    if training_file is not None:
        df_training = pd.read_csv(training_file)
        st.session_state['training_data'] = df_training
        logging.info("Added training file to session state.")

    show_data = st.checkbox('Show raw data')

# initialize session state with default parameters
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 14
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 384
if 'data_percentage' not in st.session_state:
    st.session_state.data_percentage = 1
if 'normalize_cm' not in st.session_state:
    st.session_state.normalize_cm = True
if 'min_samples_split' not in st.session_state:
    st.session_state.min_samples_split = 2
if 'min_samples_leaf' not in st.session_state:
    st.session_state.min_samples_leaf = 1
if 'max_features' not in st.session_state:
    st.session_state.max_features = 'sqrt'

# sidebar for user inputs
with st.sidebar.expander("**Model**", expanded=True):

    # user inputs

    max_depth = st.slider(
        'Max Depth', 
        min_value=1, 
        max_value=200, 
        value=st.session_state.max_depth
    )

    n_estimators = st.slider(
        'Number of Estimators', 
        min_value=1,
        max_value=1000, 
        value=st.session_state.n_estimators
    )

    min_samples_split = st.slider(
        'Min Samples Split', 
        min_value=2, 
        max_value=20, 
        value=st.session_state.min_samples_split
    )

    min_samples_leaf = st.slider(
        'Min Samples Leaf', 
        min_value=1, 
        max_value=20, 
        value=st.session_state.min_samples_leaf
    )

    max_features = st.selectbox(
        'Max Features', 
        options=['sqrt', 'log2'], 
        index=['sqrt', 'log2'].index(st.session_state.max_features)
    )

    data_percentage = st.slider(
        'Percentage of Data to Use', 
        min_value=1, 
        max_value=100, 
        value=st.session_state.data_percentage
    )

    normalize_cm = st.checkbox(
        'Normalize Confusion Matrix', 
        value=st.session_state.normalize_cm
    )

    # check if parameters have changed
    parameters_changed = (
        max_depth != st.session_state.max_depth or
        n_estimators != st.session_state.n_estimators or
        data_percentage != st.session_state.data_percentage or
        normalize_cm != st.session_state.normalize_cm or
        min_samples_split != st.session_state.min_samples_split or
        min_samples_leaf != st.session_state.min_samples_leaf or
        max_features != st.session_state.max_features
    )
    if parameters_changed:
        st.warning("Parameters have changed. New data is available. Please update.")

    # update session state
    st.session_state.max_depth = max_depth
    st.session_state.n_estimators = n_estimators
    st.session_state.data_percentage = data_percentage
    st.session_state.normalize_cm = normalize_cm
    st.session_state.min_samples_split = min_samples_split
    st.session_state.min_samples_leaf = min_samples_leaf
    st.session_state.max_features = max_features

    # update data and model
    if st.button('Update Model'):

        # spinner while loading data
        with st.spinner('Preparing the data...'):

            # load data
            load_preset_target = 'target_data' not in st.session_state
            load_preset_training = 'training_data' not in st.session_state
            st.session_state['training_data'], st.session_state['target_data'], df_combined = load_data(
                load_preset_target, 
                load_preset_training,
                st.session_state['target_data'] if not load_preset_target else None,
                st.session_state['training_data'] if not load_preset_training else None
            )
            logging.info("Data loaded successfully.")

            # prepare data
            X_train, st.session_state.X_test, y_train, st.session_state.y_test, st.session_state.label_encoder \
                = prepare_data(df_combined, data_percentage)
            logging.info("Data prepared successfully.")

        # spinner while training model
        with st.spinner('Training the model...'):
            st.session_state.rf_classifier = train_model(
                X_train, 
                y_train, 
                max_depth, 
                n_estimators, 
                min_samples_split, 
                min_samples_leaf, 
                max_features
            )
            st.session_state.y_pred = st.session_state.rf_classifier.predict(st.session_state.X_test)
            logging.info("Model trained successfully.")

        # spinner while evaluating
        with st.spinner('Evaluating the model...'):
            st.session_state.accuracy, st.session_state.precision, st.session_state.recall, \
                st.session_state.f1, st.session_state.unique_labels, st.session_state.cm = evaluate_model(
                    st.session_state.y_test, 
                    st.session_state.y_pred, 
                    st.session_state.label_encoder, 
                    st.session_state.normalize_cm
                )
        
        st.session_state.first_run = False

    # download model button
    if 'rf_classifier' in st.session_state:
        model_bytes = pickle.dumps(st.session_state.rf_classifier)
        st.download_button(
            label="Download Model",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream"
        )

# show results
if 'first_run' in st.session_state:

    # find the class with the lowest accuracy
    class_accuracies = {}
    for i, label in enumerate(st.session_state.unique_labels):
        class_y_test = (st.session_state.y_test == i).astype(int)
        class_y_pred = (st.session_state.y_pred == i).astype(int)
        class_accuracies[label] = accuracy_score(class_y_test, class_y_pred)
    st.session_state.selected_class = min(class_accuracies, key=class_accuracies.get)
    st.session_state.selected_class_index = list(st.session_state.unique_labels).index(st.session_state.selected_class)

    # allow for customization of viewing settings
    with st.sidebar.expander("**Viewer**", expanded=False):
        st.session_state.selected_class = st.selectbox(
            "Select Class", 
            st.session_state.unique_labels, 
            index=st.session_state.selected_class_index
        )

    # update selected class index when the selected class changes
    st.session_state.selected_class_index = list(st.session_state.unique_labels).index(st.session_state.selected_class)

tab1, tab2, tab3 = st.tabs([ "Data Exploration", "Explorative Error Analysis", "Feature Importance"])

# -----------------------------------------------------------
# Data Exploration
# -----------------------------------------------------------

with tab1:

    # show data as DataFrames
    if show_data:
        with st.spinner('Loading the data...'):
            df_target = st.session_state.get('target_data', pd.read_csv(preset_target))
            df_training = st.session_state.get('training_data', pd.read_csv(preset_training))
            
            with st.expander("**Training Data**", expanded=False):
                st.write(df_training)
            
            with st.expander("**Target Data**", expanded=False):
                st.write(df_target)
            
            logging.info("Raw data displayed successfully.")

    if 'first_run' not in st.session_state:
        st.warning("Please train the model first to view data exploration.")

    else:
        df_combined = pd.concat([st.session_state['training_data'], st.session_state['target_data']], axis=1)

        plot_target_distribution(df_combined)
        plot_spectral_profiles(df_combined)
        plot_wavelength_distribution(df_combined)
        
        logging.info("Data exploration displayed successfully.")

# -----------------------------------------------------------
# Explorative Error Analysis
# -----------------------------------------------------------

with tab2:

    # show placeholder
    if 'first_run' not in st.session_state:
        placeholder = st.empty()
        placeholder.warning("Please train the model first to view error analysis.")
        logging.info("Placeholder displayed successfully.")

    # show results
    if 'first_run' in st.session_state:
        visualize_error_analysis(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            st.session_state.unique_labels, 
            st.session_state.selected_class, 
            st.session_state.selected_class_index, 
            st.session_state.get('accuracy', 0), 
            st.session_state.get('precision', 0), 
            st.session_state.get('recall', 0), 
            st.session_state.get('f1', 0), 
            st.session_state.cm
        )
        logging.info("Results displayed successfully.")

# -----------------------------------------------------------
# Feature Importance & Interactions
# -----------------------------------------------------------

with tab3:
    if 'first_run' not in st.session_state:
        st.warning("Please train the model first to view feature analysis.")
    else:
        visualize_feature_importance(
            st.session_state.rf_classifier
        )
        logging.info("Feature importance displayed successfully.")
