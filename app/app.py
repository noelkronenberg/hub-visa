import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import logging
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from services.error_analysis import visualize_error_analysis
from services.feature_importance import visualize_feature_importance, visualize_interval_importance, visualize_joint_importance, get_feature_selection_inputs
from services.data import demo_cases, preset_target, preset_training, load_data, prepare_data, plot_target_distribution, plot_feature_profiles, plot_feature_distribution
from services.model import train_model, evaluate_model

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

# Initialize session state variables
if 'model1_file' not in st.session_state:
    st.session_state.model1_file = "Current Model"
if 'model2_file' not in st.session_state:
    st.session_state.model2_file = "Comparison Model"
if 'model1' not in st.session_state:
    st.session_state.model1 = None
if 'model2' not in st.session_state:
    st.session_state.model2 = None
if 'compare_models' not in st.session_state:
    st.session_state.compare_models = False
if 'rf_classifier' not in st.session_state:
    st.session_state.rf_classifier = None

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------

st.sidebar.header("Settings")

# sidebar for data exploration
with st.sidebar.expander("**Data**", expanded=False):

    # user selects the demo case
    st.session_state.selected_demo_case = st.selectbox('Select Dataset', list(demo_cases.keys()))

    # file upload
    target_file = st.file_uploader("Upload Target CSV", type=["csv"])
    training_file = st.file_uploader("Upload Training CSV", type=["csv"])

    # load data
    st.session_state['custom_target'] = False
    if target_file is not None:
        st.session_state['custom_target'] = True
        df_target = pd.read_csv(target_file)
        st.session_state['target_data'] = df_target
        logging.info("Added target file to session state.")

    # load data
    st.session_state['custom_training'] = False
    if training_file is not None:
        st.session_state['custom_training'] = True
        df_training = pd.read_csv(training_file)
        st.session_state['training_data'] = df_training
        logging.info("Added training file to session state.")

    show_data = st.checkbox('Show raw data')

    # Display current test dataset information
    st.write("**Current Test Dataset :**")

    if 'X_test' in st.session_state:
        n_features = st.session_state.X_test.shape[1]
        n_samples = st.session_state.X_test.shape[0]
        
        # Get dataset name
        if training_file is not None:
           dataset_type = training_file.name  # Use uploaded file name
        else:
            dataset_type = st.session_state.selected_demo_case if st.session_state.selected_demo_case else "Custom Dataset"
                
        st.write(f"Name: {dataset_type}")
        st.write(f"Number of features: {n_features}")
        st.write(f"Number of samples: {n_samples}")
    else:
        st.write("No test dataset loaded yet.")

# initialize session state with default parameters
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 14
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 384
if 'data_percentage' not in st.session_state:
    st.session_state.data_percentage = 10
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

    # check if the resulting data is too small or none
    if data_percentage != st.session_state.data_percentage:
        if 'training_data' in st.session_state and 'target_data' in st.session_state:
            df_combined = pd.concat([st.session_state['training_data'], st.session_state['target_data']], axis=1)
            _, _, _, _, label_encoder = prepare_data(df_combined, data_percentage)
            if label_encoder is None or len(label_encoder.classes_) == 0:
                st.warning("The resulting data is too small or none. Please choose a higher percentage.")
                logging.warning("The resulting data is too small or none. Please choose a higher percentage.")
                data_percentage = st.session_state.data_percentage

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
                target_data=st.session_state['target_data'] if not load_preset_target else None,
                training_data=st.session_state['training_data'] if not load_preset_training else None,
                selected_demo_case=st.session_state.selected_demo_case
            ) 

            if df_combined is None:
                logging.error("Data loading failed. Please adjust data settings.")
                st.error("Data loading failed. Please adjust data settings.")
            else:
                logging.info("Data loaded successfully.")

                # prepare data
                X_train, st.session_state.X_test, y_train, st.session_state.y_test, st.session_state.label_encoder \
                    = prepare_data(df_combined, data_percentage)
                logging.info("Data prepared successfully.")

        # spinner while training model
        with st.spinner('Training the model...'):

            # check if data loading failed
            try:
                # check data is not None or long enough
                if (X_train is None) or (y_train is None) or (st.session_state.X_test is None) or (st.session_state.y_test is None):
                    logging.error("Data preparation failed. Please adjust percentage of data used.")
                    st.session_state.data_error = True
                else:
                    st.session_state.data_error = False
            except NameError:
                st.session_state.data_error = True

            if st.session_state.data_error:
                pass
            else:
                # train the model
                st.session_state.rf_classifier = train_model(
                    X_train, 
                    y_train, 
                    max_depth, 
                    n_estimators, 
                    min_samples_split, 
                    min_samples_leaf, 
                    max_features
                )
                # Update model1 when rf_classifier is updated
                st.session_state.model1 = st.session_state.rf_classifier
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

    # save and download model buttons
    if 'rf_classifier' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('Save Model'):
                # Create Model directory if it doesn't exist
                model_dir = os.path.join('Model')
                os.makedirs(model_dir, exist_ok=True)
                
                existing_models = [f for f in os.listdir(model_dir) if f.startswith('model') and f.endswith('.pkl')]
                
                # Create model info dictionary
                model_info = {
                    'model': st.session_state.rf_classifier,
                    'feature_names': st.session_state.X_test.columns.tolist()
                }
                
                # Save new model
                next_model_num = len(existing_models) + 1
                model_path = os.path.join(model_dir, f'model{next_model_num}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info, f)
                st.success(f"Model saved to: {model_path}")
                logging.info(f"Model saved to {model_path}")
                
                # Refresh the list of models after saving
                existing_models = [f for f in os.listdir(model_dir) if f.startswith('model') and f.endswith('.pkl')]
                
                # Check number of models after saving
                if len(existing_models) == 3:
                    # Get creation times for all models
                    model_times = {}
                    for model in existing_models:
                        model_path = os.path.join(model_dir, model)
                        model_times[model] = os.path.getctime(model_path)
                    
                    # Find the oldest model
                    oldest_model = min(model_times, key=model_times.get)
                    
                    # Show warning when exactly 3 models are saved
                    st.warning(f"Maximum number of saved models 3 reached. The oldest model '{oldest_model}' will be automatically deleted when saving a new model.")
                    
                elif len(existing_models) >= 4:
                    # Get creation times for all models
                    model_times = {}
                    for model in existing_models:
                        model_path = os.path.join(model_dir, model)
                        model_times[model] = os.path.getctime(model_path)
                    
                    # Find and delete oldest model
                    oldest_model = min(model_times, key=model_times.get)
                    os.remove(os.path.join(model_dir, oldest_model))
                    st.info(f"Oldest model '{oldest_model}' has been deleted.")
        
            # Download button
            model_bytes = pickle.dumps(st.session_state.rf_classifier)
            st.download_button(
                label="Download Model",
                data=model_bytes,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

# -----------------------------------------------------------
# Model Comparison 
# -----------------------------------------------------------
with st.sidebar.expander("**Model Comparison**", expanded=False):
    st.write("Select models to compare")
    
    # Get all model files from the Model directory
    model_dir = os.path.join('Model')
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if len(model_files) < 2:
            st.warning("At least two saved models are required for comparison")
        else:
            # Enable model comparison checkbox
            st.session_state.compare_models = st.checkbox("Enable Model Comparison", 
                                                        key='enable_model_comparison')
            
            if st.session_state.compare_models:
                # Check if test data exists
                if 'X_test' not in st.session_state:
                    st.error("Please load test data and click 'Update Model' button first before comparing models.")
                    st.info("Steps to compare models:\n"
                           "1. Upload data or select preset dataset\n"
                           "2. Click 'Update Model' button\n"
                           "3. Then you can compare models")
                else:
                    # Select first model
                    model1_file = st.selectbox(
                        'Select Model A',
                        model_files,
                        key='model1_select'
                    )
                    
                    available_models = [f for f in model_files if f != model1_file]
                    
                    # Select second model
                    model2_file = st.selectbox(
                        'Select Model B',
                        available_models,
                        key='model2_select'
                    )
                    
                    # Load selected models
                    with open(os.path.join(model_dir, model1_file), 'rb') as f:
                        model_info = pickle.load(f)
                        st.session_state.model1 = model_info['model'] if isinstance(model_info, dict) else model_info

                    with open(os.path.join(model_dir, model2_file), 'rb') as f:
                        model_info = pickle.load(f)
                        st.session_state.model2 = model_info['model'] if isinstance(model_info, dict) else model_info
                    
                    # Save model filenames to session state
                    st.session_state.model1_file = model1_file
                    st.session_state.model2_file = model2_file
                    
                    # Set current comparison model
                    st.session_state.compare_model = st.session_state.model2
                    st.session_state.compare_model_file = model2_file

                    # Create data with correct number of features for each model
                    n_samples = len(st.session_state.X_test)

                    # Create zero matrices that fit each model
                    X_test_model1_values = np.zeros((n_samples, st.session_state.model1.n_features_in_))
                    X_test_model2_values = np.zeros((n_samples, st.session_state.model2.n_features_in_))

                    # Fill in actual data, using only available features
                    # Models trained on different datasets require different numbers of features as input when being tested
                    n_features_available = st.session_state.X_test.shape[1]
                    X_test_model1_values[:, :min(n_features_available, st.session_state.model1.n_features_in_)] = \
                        st.session_state.X_test.values[:, :min(n_features_available, st.session_state.model1.n_features_in_)]
                    X_test_model2_values[:, :min(n_features_available, st.session_state.model2.n_features_in_)] = \
                        st.session_state.X_test.values[:, :min(n_features_available, st.session_state.model2.n_features_in_)]

                    # Create DataFrame using the same feature names as during training
                    X_test_model1 = pd.DataFrame(
                        X_test_model1_values,
                        columns=[str(i) for i in range(st.session_state.model1.n_features_in_)]
                    )
                    X_test_model2 = pd.DataFrame(
                        X_test_model2_values,
                        columns=[str(i) for i in range(st.session_state.model2.n_features_in_)]
                    )

                    # Make predictions using respective test data
                    y_pred_model1 = st.session_state.model1.predict(X_test_model1)
                    y_pred_model2 = st.session_state.model2.predict(X_test_model2)
            else:
                # When comparison is disabled, allow selecting single model to display
                selected_model = st.selectbox(
                    'Select Model to Display',
                    model_files,
                    key='single_model_select'
                )
                
                # Load selected model
                with open(os.path.join(model_dir, selected_model), 'rb') as f:
                    model_info = pickle.load(f)
                    st.session_state.model1 = model_info['model'] if isinstance(model_info, dict) else model_info
                    model_name = selected_model.replace('.pkl', '')
                    st.session_state.model1_file = f"Current Model: {model_name}"
    else:
        st.warning("Model directory does not exist or is empty. Please save models first")

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

tab1, tab2, tab3 = st.tabs(["Data Exploration", "Explorative Error Analysis", "Feature Importance"])

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

    # check whether data is loaded
    elif not st.session_state.data_error:
        df_combined = pd.concat([st.session_state['training_data'], 
                              st.session_state['target_data']], axis=1)
        plot_target_distribution(df_combined)
        plot_feature_profiles(df_combined)
        plot_feature_distribution(df_combined)
        
        logging.info("Data exploration displayed successfully.")
    else:
        st.error("Data preparation failed. Full custom data are likely not provided.")
        logging.error("Data preparation failed.")

# -----------------------------------------------------------
# Explorative Error Analysis
# -----------------------------------------------------------

with tab2:
    if 'first_run' not in st.session_state or st.session_state.data_error:
        st.warning("Please train the model first to view error analysis.")
    else:
        if 'compare_models' in st.session_state and st.session_state.compare_models:
            # Use two-column layout when comparison is enabled
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{st.session_state.model1_file}**")
                visualize_error_analysis(
                    st.session_state.y_test,
                    y_pred_model1,
                    st.session_state.unique_labels,
                    st.session_state.selected_class,
                    st.session_state.selected_class_index,
                    st.session_state.accuracy,
                    st.session_state.precision,
                    st.session_state.recall,
                    st.session_state.f1,
                    st.session_state.cm
                )
            
            with col2:
                st.write(f"**{st.session_state.model2_file}**")
                if st.session_state.model2 is not None:
                    compare_accuracy, compare_precision, compare_recall, compare_f1, \
                    compare_unique_labels, compare_cm = evaluate_model(
                        st.session_state.y_test,
                        y_pred_model2,
                        st.session_state.label_encoder,
                        st.session_state.normalize_cm
                    )
                    
                    visualize_error_analysis(
                        st.session_state.y_test,
                        y_pred_model2,
                        compare_unique_labels,
                        st.session_state.selected_class,
                        st.session_state.selected_class_index,
                        compare_accuracy,
                        compare_precision,
                        compare_recall,
                        compare_f1,
                        compare_cm
                    )
                else:
                    st.warning("Model B is not loaded.")
        else:
            # Use full-width layout when no comparison
            st.write(f"**{st.session_state.model1_file}**")
            visualize_error_analysis(
                st.session_state.y_test,
                st.session_state.y_pred,
                st.session_state.unique_labels,
                st.session_state.selected_class,
                st.session_state.selected_class_index,
                st.session_state.accuracy,
                st.session_state.precision,
                st.session_state.recall,
                st.session_state.f1,
                st.session_state.cm
            )

# -----------------------------------------------------------
# Feature Importance
# -----------------------------------------------------------

with tab3:
    if 'first_run' not in st.session_state or st.session_state.data_error:
        st.warning("Please train the model first to view feature analysis.")
    else:
        if 'compare_models' in st.session_state and st.session_state.compare_models:
            if 'model1' not in st.session_state:
                st.warning("Model A is not loaded.")
            if 'model2' not in st.session_state:
                st.warning("Model B is not loaded.")
            
            if 'model1' in st.session_state and 'model2' in st.session_state:
                # Display feature importance for both models
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Feature Importance - {st.session_state.model1_file}")
                    visualize_feature_importance(st.session_state.model1, key_suffix='model1')
                with col2:
                    st.write(f"Feature Importance - {st.session_state.model2_file}")
                    visualize_feature_importance(st.session_state.model2, key_suffix='model2')
        else:
            if 'model1' not in st.session_state:
                st.warning("Model is not loaded.")
            else:
                # Display feature importance for single model
                st.write("Feature Importance")
                visualize_feature_importance(st.session_state.model1, key_suffix='single')

        with st.expander("**Interval Importance**", expanded=False):

            st.write("""
                Assess the impact of feature value intervals on the prediction accuracy by splitting a feature into intervals and mapping every data point to the boundaries of that interval. By comparing evaluation metrics of original data to the one with a transformed interval of our choice, we derive the importance of that interval to the prediction.
            """)

            # get feature importances
            importances = st.session_state.rf_classifier.feature_importances_
            feature_names = st.session_state.X_test.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            # preselect feature with highest importance
            selected_feature = st.selectbox(
                "Select Feature", 
                feature_importance_df['feature'], 
                index=0
            )
            logging.info(f"Feature selected successfully: {selected_feature}")

            # get index of selected feature
            feature_index = list(st.session_state.X_test.columns).index(selected_feature)
            logging.info(f"Feature index: {feature_index}")
        
            # ask user how many intervals to define
            num_intervals = st.slider(
                'Number of Intervals', 
                min_value=1, 
                max_value=20, 
                value=10
            )
            logging.info(f"CHose number of intervals: {num_intervals}")

            # plot interval importance
            visualize_interval_importance(
                st.session_state.rf_classifier,
                st.session_state.X_test, 
                st.session_state.y_test, 
                st.session_state.accuracy, 
                st.session_state.precision, 
                st.session_state.recall, 
                st.session_state.f1,
                feature_index, 
                num_intervals
            )

        # joint interval importance
        with st.expander("**Joint Interval Importance**", expanded=False):

            st.write("""
                Assess the impact of feature value intervals on the prediction accuracy by splitting two features into intervals and mapping every data point to the boundaries of the intervals. By comparing evaluation metrics of original data to the ones with transformed intervals of our choice, we derive the importance of the intervals to the prediction.
            """)

            # get feature importance values
            importances = st.session_state.rf_classifier.feature_importances_
            feature_names = st.session_state.X_test.columns
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            # get feature selection inputs from user
            selected_feature1, selected_feature2, feature1_index, feature2_index, num_intervals1, num_intervals2 = \
                get_feature_selection_inputs(feature_importance_df, feature_names)

            # visualize joint importance
            visualize_joint_importance(
                st.session_state.rf_classifier,
                st.session_state.X_test,
                st.session_state.y_test,
                feature1_index,
                feature2_index,
                num_intervals1,
                num_intervals2
            )
