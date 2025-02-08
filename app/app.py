import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import logging
import pickle

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

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------

st.sidebar.header("Settings")

with st.sidebar.expander("**Model Comparison**", expanded=False):
    # toggle compairson model
    st.write("Train a comparison model to compare the results of the main model with another model.")
    st.session_state.train_comparison_model = st.checkbox("Train Comparison Model", value=False)
    # determine suffix based on comparison model
    st.session_state.suffix = '_compare' if st.session_state.get('train_comparison_model', False) else ''
    # toggle compare models
    st.session_state.compare_models = False
    if 'rf_classifier' in st.session_state or 'rf_classifier_compare' in st.session_state:
        st.session_state.compare_models = st.checkbox("Compare Models", value=False)

# sidebar for data exploration
with st.sidebar.expander("**Data**", expanded=False):

    # user selects the demo case
    selected_demo_case_key = f'selected_demo_case{st.session_state.suffix}'
    selected_demo_case = st.session_state.get(selected_demo_case_key, list(demo_cases.keys())[0])
    st.session_state[selected_demo_case_key] = st.selectbox('Select Dataset', list(demo_cases.keys()), index=list(demo_cases.keys()).index(selected_demo_case))

    # file upload
    target_file = st.file_uploader("Upload Target CSV", type=["csv"])
    training_file = st.file_uploader("Upload Training CSV", type=["csv"])

    st.session_state[f'custom_target{st.session_state.suffix}'] = target_file is not None
    if target_file is not None:
        df_target = pd.read_csv(target_file)
        st.session_state[f'target_data{st.session_state.suffix}'] = df_target
        logging.info(f"Added target file {'for comparison model ' if st.session_state.train_comparison_model else ''}to session state.")

    st.session_state[f'custom_training{st.session_state.suffix}'] = training_file is not None
    if training_file is not None:
        df_training = pd.read_csv(training_file)
        st.session_state[f'training_data{st.session_state.suffix}'] = df_training
        logging.info(f"Added training file {'for comparison model ' if st.session_state.train_comparison_model else ''}to session state.")

    show_data = st.checkbox('Show raw data')

# initialize session state with default parameters

defaults = {
    'max_depth': 14,
    'n_estimators': 384,
    'data_percentage': 10,
    'normalize_cm': True,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}

for key, value in defaults.items():
        session_key = f"{key}{st.session_state.suffix}"
        if session_key not in st.session_state:
            st.session_state[session_key] = value

# sidebar for user inputs
with st.sidebar.expander("**Model**", expanded=True):

    # user inputs
    max_depth = st.slider(
        'Max Depth', 
        min_value=1, 
        max_value=200, 
        value=st.session_state[f'max_depth{st.session_state.suffix}']
    )

    n_estimators = st.slider(
        'Number of Estimators', 
        min_value=1,
        max_value=1000, 
        value=st.session_state[f'n_estimators{st.session_state.suffix}']
    )

    min_samples_split = st.slider(
        'Min Samples Split', 
        min_value=2, 
        max_value=20, 
        value=st.session_state[f'min_samples_split{st.session_state.suffix}']
    )

    min_samples_leaf = st.slider(
        'Min Samples Leaf', 
        min_value=1, 
        max_value=20, 
        value=st.session_state[f'min_samples_leaf{st.session_state.suffix}']
    )

    max_features = st.selectbox(
        'Max Features', 
        options=['sqrt', 'log2'], 
        index=['sqrt', 'log2'].index(st.session_state[f'max_features{st.session_state.suffix}'])
    )

    data_percentage = st.slider(
        'Percentage of Data to Use', 
        min_value=1, 
        max_value=100, 
        value=st.session_state[f'data_percentage{st.session_state.suffix}']
    )

    # check if the resulting data is too small or none
    if data_percentage != st.session_state[f'data_percentage{st.session_state.suffix}']:
        if 'training_data' in st.session_state and 'target_data' in st.session_state:
            df_combined = pd.concat([st.session_state['training_data'], st.session_state['target_data']], axis=1)
            _, _, _, _, label_encoder = prepare_data(df_combined, data_percentage)
            if label_encoder is None or len(label_encoder.classes_) == 0:
                st.warning("The resulting data is too small or none. Please choose a higher percentage.")
                logging.warning("The resulting data is too small or none. Please choose a higher percentage.")
                data_percentage = st.session_state[f'data_percentage{st.session_state.suffix}']

    normalize_cm = st.checkbox(
        'Normalize Confusion Matrix', 
        value=st.session_state[f'normalize_cm{st.session_state.suffix}']
    )

    # check if parameters have changed
    parameters_changed = (
        max_depth != st.session_state[f'max_depth{st.session_state.suffix}'] or
        n_estimators != st.session_state[f'n_estimators{st.session_state.suffix}'] or
        data_percentage != st.session_state[f'data_percentage{st.session_state.suffix}'] or
        normalize_cm != st.session_state[f'normalize_cm{st.session_state.suffix}'] or
        min_samples_split != st.session_state[f'min_samples_split{st.session_state.suffix}'] or
        min_samples_leaf != st.session_state[f'min_samples_leaf{st.session_state.suffix}'] or
        max_features != st.session_state[f'max_features{st.session_state.suffix}']
    )
    if parameters_changed:
        st.warning("Parameters have changed. New data is available. Please update.")

    # update session state
    st.session_state[f'max_depth{st.session_state.suffix}'] = max_depth
    st.session_state[f'n_estimators{st.session_state.suffix}'] = n_estimators
    st.session_state[f'data_percentage{st.session_state.suffix}'] = data_percentage
    st.session_state[f'normalize_cm{st.session_state.suffix}'] = normalize_cm
    st.session_state[f'min_samples_split{st.session_state.suffix}'] = min_samples_split
    st.session_state[f'min_samples_leaf{st.session_state.suffix}'] = min_samples_leaf
    st.session_state[f'max_features{st.session_state.suffix}'] = max_features
    # update data and model
    if st.button('Update Model'):

        # spinner while loading data
        with st.spinner('Preparing the data...'):

            # load data
            load_preset_target = f'target_data{st.session_state.suffix}' not in st.session_state
            load_preset_training = f'training_data{st.session_state.suffix}' not in st.session_state
            st.session_state[f'training_data{st.session_state.suffix}'], st.session_state[f'target_data{st.session_state.suffix}'], df_combined = load_data(
                target_data=st.session_state[f'target_data{st.session_state.suffix}'] if not load_preset_target else None,
                training_data=st.session_state[f'training_data{st.session_state.suffix}'] if not load_preset_training else None,
                selected_demo_case=st.session_state[f'selected_demo_case{st.session_state.suffix}']
            ) 

            if df_combined is None:
                logging.error("Data loading failed. Please adjust data settings.")
                st.error("Data loading failed. Please adjust data settings.")
            else:
                logging.info("Data loaded successfully.")

                # prepare data
                X_train, st.session_state[f'X_test{st.session_state.suffix}'], y_train, st.session_state[f'y_test{st.session_state.suffix}'], st.session_state[f'label_encoder{st.session_state.suffix}'] \
                    = prepare_data(df_combined, data_percentage)
                logging.info("Data prepared successfully.")

        # spinner while training model
        with st.spinner('Training the model...'):

            # check if data loading failed
            try:
                # check data is not None or long enough
                if (X_train is None) or (y_train is None) or (st.session_state[f'X_test{st.session_state.suffix}'] is None) or (st.session_state[f'y_test{st.session_state.suffix}'] is None):
                    logging.error("Data preparation failed. Please adjust percentage of data used.")
                    st.session_state[f'data_error{st.session_state.suffix}'] = True
                else:
                    st.session_state[f'data_error{st.session_state.suffix}'] = False
            except NameError:
                st.session_state[f'data_error{st.session_state.suffix}'] = True

            if st.session_state[f'data_error{st.session_state.suffix}']:
                pass
            else:
                # train the model
                st.session_state[f'rf_classifier{st.session_state.suffix}'] = train_model(
                    X_train, 
                    y_train, 
                    max_depth, 
                    n_estimators, 
                    min_samples_split, 
                    min_samples_leaf, 
                    max_features
                )
                st.session_state[f'y_pred{st.session_state.suffix}'] = st.session_state[f'rf_classifier{st.session_state.suffix}'].predict(st.session_state[f'X_test{st.session_state.suffix}'])
                logging.info("Model trained successfully.")

                # spinner while evaluating
                with st.spinner('Evaluating the model...'):
                    st.session_state[f'accuracy{st.session_state.suffix}'], st.session_state[f'precision{st.session_state.suffix}'], st.session_state[f'recall{st.session_state.suffix}'], \
                        st.session_state[f'f1{st.session_state.suffix}'], st.session_state[f'unique_labels{st.session_state.suffix}'], st.session_state[f'cm{st.session_state.suffix}'] = evaluate_model(
                            st.session_state[f'y_test{st.session_state.suffix}'], 
                            st.session_state[f'y_pred{st.session_state.suffix}'], 
                            st.session_state[f'label_encoder{st.session_state.suffix}'], 
                            st.session_state[f'normalize_cm{st.session_state.suffix}']
                        )
                
                st.session_state[f'first_run{st.session_state.suffix}'] = False

    # download model button
    if f'rf_classifier{st.session_state.suffix}' in st.session_state:
        model_bytes = pickle.dumps(st.session_state[f'rf_classifier{st.session_state.suffix}'])
        st.download_button(
            label="Download Model",
            data=model_bytes,
            file_name=f"trained_model{st.session_state.suffix}.pkl",
            mime="application/octet-stream"
        )

# show results
if f'first_run{st.session_state.suffix}' in st.session_state:

    # find the class with the lowest accuracy
    class_accuracies = {}
    for i, label in enumerate(st.session_state[f'unique_labels{st.session_state.suffix}']):
        class_y_test = (st.session_state[f'y_test{st.session_state.suffix}'] == i).astype(int)
        class_y_pred = (st.session_state[f'y_pred{st.session_state.suffix}'] == i).astype(int)
        class_accuracies[label] = accuracy_score(class_y_test, class_y_pred)
    st.session_state[f'selected_class{st.session_state.suffix}'] = min(class_accuracies, key=class_accuracies.get)
    st.session_state[f'selected_class_index{st.session_state.suffix}'] = list(st.session_state[f'unique_labels{st.session_state.suffix}']).index(st.session_state[f'selected_class{st.session_state.suffix}'])

    # allow for customization of viewing settings
    with st.sidebar.expander("**Viewer**", expanded=False):
        st.session_state[f'selected_class{st.session_state.suffix}'] = st.selectbox(
            "Select Class", 
            st.session_state[f'unique_labels{st.session_state.suffix}'], 
            index=st.session_state[f'selected_class_index{st.session_state.suffix}']
        )

    # update selected class index when the selected class changes
    st.session_state[f'selected_class_index{st.session_state.suffix}'] = list(st.session_state[f'unique_labels{st.session_state.suffix}']).index(st.session_state[f'selected_class{st.session_state.suffix}'])

tab1, tab2, tab3 = st.tabs([ "Data Exploration", "Explorative Error Analysis", "Feature Importance"])

# -----------------------------------------------------------
# Data Exploration
# -----------------------------------------------------------

with tab1:

    # show data as DataFrames
    if show_data:
        with st.spinner('Loading the data...'):
            df_target = st.session_state.get(f'target_data{st.session_state.suffix}', pd.read_csv(preset_target))
            df_training = st.session_state.get(f'training_data{st.session_state.suffix}', pd.read_csv(preset_training))
            
            with st.expander("**Training Data**", expanded=False):
                st.write(df_training)
            
            with st.expander("**Target Data**", expanded=False):
                st.write(df_target)
            
            logging.info("Raw data displayed successfully.")

    if f'first_run{st.session_state.suffix}' not in st.session_state:
        st.warning("Please train the model first to view data exploration.")

    # check whether data is loaded
    elif not st.session_state.data_error:
        if st.session_state.compare_models:
            col1, col2 = st.columns(2)

            with col1:
                df_combined = pd.concat([st.session_state[f'training_data'], st.session_state[f'target_data']], axis=1)
                st.subheader("Main Model")
                plot_target_distribution(df_combined, suffix='')
                plot_feature_profiles(df_combined, suffix='')
                plot_feature_distribution(df_combined, suffix='')
                logging.info("Main model data exploration displayed successfully.")

            with col2:
                st.subheader("Comparison Model")
                df_combined_compare = pd.concat([st.session_state[f'training_data_compare'], st.session_state[f'target_data_compare']], axis=1)
                plot_target_distribution(df_combined_compare, suffix='_compare')
                plot_feature_profiles(df_combined_compare, suffix='_compare')
                plot_feature_distribution(df_combined_compare, suffix='_compare')
                logging.info("Comparison model data exploration displayed successfully.")
        else:
            df_combined = pd.concat([st.session_state[f'training_data{st.session_state.suffix}'], st.session_state[f'target_data{st.session_state.suffix}']], axis=1)
            plot_target_distribution(df_combined, suffix='')
            plot_feature_profiles(df_combined, suffix='')
            plot_feature_distribution(df_combined, suffix='')
            logging.info("Data exploration displayed successfully.")
    else:
        st.error("Data preparation failed. Full custom data are likely not provided. Please upload both target and training data.")
        logging.error("Data preparation failed. Full custom data are likely not provided. Please upload both target and training data.")

# -----------------------------------------------------------
# Explorative Error Analysis
# -----------------------------------------------------------

with tab2:

    if f'first_run{st.session_state.suffix}' not in st.session_state or st.session_state[f'data_error{st.session_state.suffix}']:
        placeholder = st.empty()
        placeholder.warning("Please train the model first to view error analysis.")
        logging.info("Placeholder displayed successfully.")
    else:
        if st.session_state.compare_models:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Main Model")
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
                    st.session_state.cm,
                    suffix=''
                )
                logging.info("Main model results displayed successfully.")

            with col2:
                st.subheader("Comparison Model")
                visualize_error_analysis(
                    st.session_state[f'y_test_compare'], 
                    st.session_state[f'y_pred_compare'], 
                    st.session_state[f'unique_labels_compare'], 
                    st.session_state[f'selected_class_compare'], 
                    st.session_state[f'selected_class_index_compare'], 
                    st.session_state.get(f'accuracy_compare', 0), 
                    st.session_state.get(f'precision_compare', 0), 
                    st.session_state.get(f'recall_compare', 0), 
                    st.session_state.get(f'f1_compare', 0), 
                    st.session_state[f'cm_compare'],
                    suffix='_compare'
                )
                logging.info("Comparison model results displayed successfully.")
        else:
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
                st.session_state.cm,
                suffix=''
            )
            logging.info("Results displayed successfully.")

# -----------------------------------------------------------
# Feature Importance & Interactions
# -----------------------------------------------------------

with tab3:
    if f'first_run{st.session_state.suffix}' not in st.session_state or st.session_state[f'data_error{st.session_state.suffix}']:
        st.warning("Please train the model first to view feature analysis.")
    else:
        if st.session_state.compare_models:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Main Model")
                with st.expander("**Feature Importance**", expanded=True):
                    visualize_feature_importance(
                        st.session_state.rf_classifier,
                        suffix=''
                    )
                    logging.info("Feature importance displayed successfully.")

                with st.expander("**Interval Importance**", expanded=False):
                    st.write("""
                        Assess the impact of feature value intervals on the prediction accuracy by splitting a feature into intervals and mapping every data point to the boundaries of that interval. By comparing evaluation metrics of original data to the one with a transformed interval of our choice, we derive the importance of that interval to the prediction.
                    """)
                    importances = st.session_state.rf_classifier.feature_importances_
                    feature_names = st.session_state.X_test.columns
                    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                    selected_feature = st.selectbox("Select Feature", feature_importance_df['feature'], index=0, key='feature')
                    feature_index = list(st.session_state.X_test.columns).index(selected_feature)
                    num_intervals = st.slider('Number of Intervals', min_value=1, max_value=20, value=10, key='intervals')
                    visualize_interval_importance(
                        st.session_state.rf_classifier, 
                        st.session_state.X_test, 
                        st.session_state.y_test, 
                        st.session_state.accuracy, 
                        st.session_state.precision, 
                        st.session_state.recall, 
                        st.session_state.f1, 
                        feature_index, 
                        num_intervals,
                        suffix=''
                    )

                with st.expander("**Joint Interval Importance**", expanded=False):
                    st.write("""
                        Assess the impact of feature value intervals on the prediction accuracy by splitting two features into intervals and mapping every data point to the boundaries of the intervals. By comparing evaluation metrics of original data to the ones with transformed intervals of our choice, we derive the importance of the intervals to the prediction.
                    """)
                    importances = st.session_state.rf_classifier.feature_importances_
                    feature_names = st.session_state.X_test.columns
                    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                    selected_feature1, selected_feature2, feature1_index, feature2_index, num_intervals1, num_intervals2 = get_feature_selection_inputs(feature_importance_df, feature_names, suffix='')
                    visualize_joint_importance(
                        st.session_state.rf_classifier, 
                        st.session_state.X_test, 
                        st.session_state.y_test, 
                        feature1_index, 
                        feature2_index, 
                        num_intervals1, 
                        num_intervals2,
                        suffix=''
                    )

            with col2:
                st.subheader("Comparison Model")
                with st.expander("**Feature Importance**", expanded=True):
                    visualize_feature_importance(
                        st.session_state[f'rf_classifier{st.session_state.suffix}'],
                        suffix='_compare'
                    )
                    logging.info("Feature importance displayed successfully.")

                with st.expander("**Interval Importance**", expanded=False):
                    st.write("""
                        Assess the impact of feature value intervals on the prediction accuracy by splitting a feature into intervals and mapping every data point to the boundaries of that interval. By comparing evaluation metrics of original data to the one with a transformed interval of our choice, we derive the importance of that interval to the prediction.
                    """)
                    importances = st.session_state[f'rf_classifier{st.session_state.suffix}'].feature_importances_
                    feature_names = st.session_state[f'X_test{st.session_state.suffix}'].columns
                    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                    selected_feature = st.selectbox("Select Feature", feature_importance_df['feature'], index=0, key='feature_compare')
                    feature_index = list(st.session_state[f'X_test{st.session_state.suffix}'].columns).index(selected_feature)
                    num_intervals = st.slider('Number of Intervals', min_value=1, max_value=20, value=10, key='intervals_compare')
                    visualize_interval_importance(
                        st.session_state[f'rf_classifier{st.session_state.suffix}'], 
                        st.session_state[f'X_test{st.session_state.suffix}'], 
                        st.session_state[f'y_test{st.session_state.suffix}'], 
                        st.session_state[f'accuracy{st.session_state.suffix}'], 
                        st.session_state[f'precision{st.session_state.suffix}'], 
                        st.session_state[f'recall{st.session_state.suffix}'], 
                        st.session_state[f'f1{st.session_state.suffix}'], 
                        feature_index, 
                        num_intervals,
                        suffix='_compare'
                    )

                with st.expander("**Joint Interval Importance**", expanded=False):
                    st.write("""
                        Assess the impact of feature value intervals on the prediction accuracy by splitting two features into intervals and mapping every data point to the boundaries of the intervals. By comparing evaluation metrics of original data to the ones with transformed intervals of our choice, we derive the importance of the intervals to the prediction.
                    """)
                    importances = st.session_state[f'rf_classifier{st.session_state.suffix}'].feature_importances_
                    feature_names = st.session_state[f'X_test{st.session_state.suffix}'].columns
                    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                    selected_feature1, selected_feature2, feature1_index, feature2_index, num_intervals1, num_intervals2 = get_feature_selection_inputs(feature_importance_df, feature_names, suffix='_compare')
                    visualize_joint_importance(
                        st.session_state[f'rf_classifier{st.session_state.suffix}'], 
                        st.session_state[f'X_test{st.session_state.suffix}'], 
                        st.session_state[f'y_test{st.session_state.suffix}'], 
                        feature1_index, 
                        feature2_index, 
                        num_intervals1, 
                        num_intervals2,
                        suffix='_compare'
                    )
        else:
            with st.expander("**Feature Importance**", expanded=True):
                visualize_feature_importance(
                    st.session_state.rf_classifier,
                    suffix=''
                )
                logging.info("Feature importance displayed successfully.")

            with st.expander("**Interval Importance**", expanded=False):
                st.write("""
                    Assess the impact of feature value intervals on the prediction accuracy by splitting a feature into intervals and mapping every data point to the boundaries of that interval. By comparing evaluation metrics of original data to the one with a transformed interval of our choice, we derive the importance of that interval to the prediction.
                """)
                importances = st.session_state.rf_classifier.feature_importances_
                feature_names = st.session_state.X_test.columns
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                selected_feature = st.selectbox("Select Feature", feature_importance_df['feature'], index=0, key=f'feature')
                feature_index = list(st.session_state.X_test.columns).index(selected_feature)
                num_intervals = st.slider('Number of Intervals', min_value=1, max_value=20, value=10, key=f'intervals')
                visualize_interval_importance(
                    st.session_state.rf_classifier, 
                    st.session_state.X_test, 
                    st.session_state.y_test, 
                    st.session_state.accuracy, 
                    st.session_state.precision, 
                    st.session_state.recall, 
                    st.session_state.f1, 
                    feature_index, 
                    num_intervals,
                    suffix=''
                )

            with st.expander("**Joint Interval Importance**", expanded=False):
                st.write("""
                    Assess the impact of feature value intervals on the prediction accuracy by splitting two features into intervals and mapping every data point to the boundaries of the intervals. By comparing evaluation metrics of original data to the ones with transformed intervals of our choice, we derive the importance of the intervals to the prediction.
                """)
                importances = st.session_state.rf_classifier.feature_importances_
                feature_names = st.session_state.X_test.columns
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                selected_feature1, selected_feature2, feature1_index, feature2_index, num_intervals1, num_intervals2 = get_feature_selection_inputs(feature_importance_df, feature_names, suffix='')
                visualize_joint_importance(
                    st.session_state.rf_classifier, 
                    st.session_state.X_test, 
                    st.session_state.y_test, 
                    feature1_index, 
                    feature2_index, 
                    num_intervals1, 
                    num_intervals2,
                    suffix=''
                )
