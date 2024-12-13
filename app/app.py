import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import logging

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# streamlit app
st.set_page_config(page_title="VisA Dashboard", layout="wide")
st.title("VisA Dashboard")

# sidebar for data exploration
st.sidebar.header("Data Settings")

# preset data
preset_target = 'app/lucas_organic_carbon/target/lucas_organic_carbon_target.csv'
preset_training = 'app/lucas_organic_carbon/training_test/compressed_data.csv'

# file upload
target_file = st.sidebar.file_uploader("Upload Target CSV", type=["csv"])
training_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])

if target_file is not None:
    df_target = pd.read_csv(target_file)
    st.session_state['target_data'] = df_target
    logging.info("Added target file to session state.")

if training_file is not None:
    df_training = pd.read_csv(training_file)
    st.session_state['training_data'] = df_training
    logging.info("Added training file to session state.")

if st.sidebar.checkbox('Show raw data'):
    df_target = st.session_state.get('target_data', pd.read_csv(preset_target))
    df_training = st.session_state.get('training_data', pd.read_csv(preset_training))
    st.subheader('Training Data')
    st.write(df_training)
    st.subheader('Target Data')
    st.write(df_target)
    logging.info("Raw data displayed successfully.")

# initialize session state with default parameters
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 14
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 384
if 'data_percentage' not in st.session_state:
    st.session_state.data_percentage = 1
if 'normalize_cm' not in st.session_state:
    st.session_state.normalize_cm = True

# sidebar for user inputs
st.sidebar.header("Model Settings")
max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=200, value=st.session_state.max_depth)
n_estimators = st.sidebar.slider('Number of Estimators', min_value=1, max_value=1000, value=st.session_state.n_estimators)
data_percentage = st.sidebar.slider('Percentage of Data to Use', min_value=1, max_value=100, value=st.session_state.data_percentage)
normalize_cm = st.sidebar.checkbox('Normalize Confusion Matrix', value=st.session_state.normalize_cm)

# check if parameters have changed
parameters_changed = (
    max_depth != st.session_state.max_depth or
    n_estimators != st.session_state.n_estimators or
    data_percentage != st.session_state.data_percentage or
    normalize_cm != st.session_state.normalize_cm
)
if parameters_changed:
    st.sidebar.warning("Parameters have changed. New data is available. Please update.")

# update session state
st.session_state.max_depth = max_depth
st.session_state.n_estimators = n_estimators
st.session_state.data_percentage = data_percentage
st.session_state.normalize_cm = normalize_cm

# load data
@st.cache_data(show_spinner=False)
def load_data():

    # load data if not in session state

    if 'target_data' not in st.session_state:
        df_target = pd.read_csv(preset_target)
        st.session_state['target_data'] = df_target
        logging.info("Loaded target data from preset.")
    else:
        df_target = st.session_state['target_data']
        logging.info("Loaded target data from uploaded file.")

    if 'training_data' not in st.session_state:
        df_training = pd.read_csv(preset_training) # NOTE: this is a compressed version of the data (for demo purposes)
        st.session_state['training_data'] = df_training
        logging.info("Loaded training data from preset.")
    else:
        df_training = st.session_state['training_data']
        logging.info("Loaded training data from uploaded file.")
    
    df_combined = pd.merge(df_training, df_target, left_index=True, right_index=True)
    
    return df_combined

# prepare data
@st.cache_data(show_spinner=False)
def prepare_data(df_combined, data_percentage):
    # sample the data
    sample_size = int(len(df_combined) * (data_percentage / 100))
    df_sampled = df_combined.sample(n=sample_size, random_state=42)
    predictors = df_sampled.columns[:-1]

    # prepare the data
    target = df_sampled.columns[-1]
    X = df_sampled[predictors]
    y = df_sampled[target]

    # encode categorical target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

# train model
@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train, max_depth, n_estimators):
    rf_classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# update data and model
if st.sidebar.button('Update'):

    # spinner while loading data
    with st.spinner('Preparing the data...'):
        df_combined = load_data()
        X_train, st.session_state.X_test, y_train, st.session_state.y_test, label_encoder = prepare_data(df_combined, data_percentage)
        logging.info("Data prepared successfully.")

    # spinner while training model
    with st.spinner('Training the model...'):
        rf_classifier = train_model(X_train, y_train, max_depth, n_estimators)
        st.session_state.y_pred = rf_classifier.predict(st.session_state.X_test)
        logging.info("Model trained successfully.")

    # spinner while evaluating
    with st.spinner('Evaluating the model...'):

        # compute metrics (and save in session state)
        st.session_state.accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        st.session_state.precision = precision_score(st.session_state.y_test, st.session_state.y_pred, average='weighted')
        st.session_state.recall = recall_score(st.session_state.y_test, st.session_state.y_pred, average='weighted')
        st.session_state.f1 = f1_score(st.session_state.y_test, st.session_state.y_pred, average='weighted')

        # actual label names (and save in session state)
        st.session_state.unique_labels = label_encoder.classes_

        # compute confusion matrix (and save in session state)
        if normalize_cm:
            st.session_state.cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred, labels=range(len(st.session_state.unique_labels)), normalize='true')
        else:
            st.session_state.cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred, labels=range(len(st.session_state.unique_labels)))

        logging.info("Model evaluated successfully.")
    
    st.session_state.first_run = False

# show results
def visualize():

    # display metrics (in columns)

    # overall metrics 
    st.subheader("Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{st.session_state.get('accuracy', 0):.2f}")
    col2.metric("Precision", f"{st.session_state.get('precision', 0):.2f}")
    col3.metric("Recall", f"{st.session_state.get('recall', 0):.2f}")
    col4.metric("F1 Score", f"{st.session_state.get('f1', 0):.2f}")
    logging.info("Metrics displayed successfully.")

    # metrics for selected class

    class_y_test = (st.session_state.y_test == st.session_state.selected_class_index).astype(int)
    class_y_pred = (st.session_state.y_pred == st.session_state.selected_class_index).astype(int)
    class_accuracy = accuracy_score(class_y_test, class_y_pred)
    class_precision = precision_score(class_y_test, class_y_pred)
    class_recall = recall_score(class_y_test, class_y_pred)
    class_f1 = f1_score(class_y_test, class_y_pred)

    st.subheader(f"Metrics for Class: {st.session_state.selected_class}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{class_accuracy:.2f}")
    col2.metric("Precision", f"{class_precision:.2f}")
    col3.metric("Recall", f"{class_recall:.2f}")
    col4.metric("F1 Score", f"{class_f1:.2f}")
    logging.info(f"Metrics for class {st.session_state.selected_class} displayed successfully.")

    # display confusion matrix
    st.subheader('Confusion Matrix')
    fig = go.Figure(data=go.Heatmap(
        z=st.session_state.cm,
        x=[f'Predicted: {label}' for label in st.session_state.unique_labels],
        y=[f'Actual: {label}' for label in st.session_state.unique_labels],
        colorscale='Viridis',
        showscale=True
    ))
    fig.update_layout(
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        xaxis=dict(tickmode='array', tickvals=list(range(len(st.session_state.unique_labels))), ticktext=st.session_state.unique_labels),
        yaxis=dict(tickmode='array', tickvals=list(range(len(st.session_state.unique_labels))), ticktext=st.session_state.unique_labels)
    )
    # add rectangles to highlight the selected class (row and column)
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=len(st.session_state.unique_labels) - 0.5,
        y0=st.session_state.selected_class_index - 0.5, y1=st.session_state.selected_class_index + 0.5,
        line=dict(color="red", width=3)
    )
    fig.add_shape(
        type="rect",
        x0=st.session_state.selected_class_index - 0.5, x1=st.session_state.selected_class_index + 0.5,
        y0=-0.5, y1=len(st.session_state.unique_labels) - 0.5,
        line=dict(color="red", width=3)
    )
    st.plotly_chart(fig)
    logging.info("Confusion matrix displayed successfully.")

if __name__ == "__main__":

    # show placeholder
    if 'first_run' not in st.session_state:
        placeholder = st.empty()
        placeholder.warning("Adjust the parameters and run the model to view results.")
        logging.info("Placeholder displayed successfully.")

    # show results
    if 'first_run' in st.session_state:

        # display metrics for selected class
        st.sidebar.header("Class Metrics")
        st.session_state.selected_class = st.sidebar.selectbox("Select Class", st.session_state.unique_labels)
        st.session_state.selected_class_index = list(st.session_state.unique_labels).index(st.session_state.selected_class)

        visualize()
        logging.info("Results displayed successfully.")