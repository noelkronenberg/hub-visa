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

st.sidebar.header("Settings")

# sidebar for data exploration
with st.sidebar.expander("**Data**", expanded=False):
    # preset data
    preset_target = 'app/lucas_organic_carbon/target/lucas_organic_carbon_target.csv'
    preset_training = 'app/lucas_organic_carbon/training_test/compressed_data.csv'

    # file upload
    target_file = st.file_uploader("Upload Target CSV", type=["csv"])
    training_file = st.file_uploader("Upload Training CSV", type=["csv"])

    if target_file is not None:
        df_target = pd.read_csv(target_file)
        st.session_state['target_data'] = df_target
        logging.info("Added target file to session state.")

    if training_file is not None:
        df_training = pd.read_csv(training_file)
        st.session_state['training_data'] = df_training
        logging.info("Added training file to session state.")
    
    show_data = st.checkbox('Show raw data')

if show_data:
    with st.spinner('Loading the data...'):
        df_target = st.session_state.get('target_data', pd.read_csv(preset_target))
        df_training = st.session_state.get('training_data', pd.read_csv(preset_training))
        
        with st.expander("**Training Data**", expanded=False):
            st.write(df_training)
        
        with st.expander("**Target Data**", expanded=False):
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
if 'min_samples_split' not in st.session_state:
    st.session_state.min_samples_split = 2
if 'min_samples_leaf' not in st.session_state:
    st.session_state.min_samples_leaf = 1
if 'max_features' not in st.session_state:
    st.session_state.max_features = 'sqrt'

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
        df_training = pd.read_csv(preset_training)  # NOTE: this is a compressed version of the data (for demo purposes)
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
def train_model(X_train, y_train, max_depth, n_estimators, min_samples_split, min_samples_leaf, max_features):
    rf_classifier = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# sidebar for user inputs
with st.sidebar.expander("**Model**", expanded=True):
    
    max_depth = st.slider('Max Depth', min_value=1, max_value=200, value=st.session_state.max_depth)
    n_estimators = st.slider('Number of Estimators', min_value=1, max_value=1000, value=st.session_state.n_estimators)
    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=st.session_state.min_samples_split)
    min_samples_leaf = st.slider('Min Samples Leaf', min_value=1, max_value=20, value=st.session_state.min_samples_leaf)
    max_features = st.selectbox('Max Features', options=['sqrt', 'log2'], index=['sqrt', 'log2'].index(st.session_state.max_features))

    data_percentage = st.slider('Percentage of Data to Use', min_value=1, max_value=100, value=st.session_state.data_percentage)
    normalize_cm = st.checkbox('Normalize Confusion Matrix', value=st.session_state.normalize_cm)

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
            df_combined = load_data()
            X_train, st.session_state.X_test, y_train, st.session_state.y_test, label_encoder = prepare_data(df_combined, data_percentage)
            logging.info("Data prepared successfully.")

        # spinner while training model
        with st.spinner('Training the model...'):
            rf_classifier = train_model(X_train, y_train, max_depth, n_estimators, min_samples_split, min_samples_leaf, max_features)
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

def display_class_counts(y_test, y_pred, unique_labels):

    # count instances of each class in the test set
    actual_class_counts = pd.Series(y_test).value_counts().reindex(range(len(unique_labels)), fill_value=0)
    predicted_class_counts = pd.Series(y_pred).value_counts().reindex(range(len(unique_labels)), fill_value=0)

    # sort by count
    sorted_counts = actual_class_counts.sort_values(ascending=False)
    sorted_labels = [unique_labels[i] for i in sorted_counts.index]

    # create a bar chart
    fig = go.Figure(data=[
        go.Bar(
            name='Actual',
            x=sorted_labels, 
            y=sorted_counts.values, 
            marker_color=['red' if label == st.session_state.selected_class else 'black' for label in sorted_labels]
        ),
        go.Bar(
            name='Predicted',
            x=sorted_labels, 
            y=predicted_class_counts[sorted_counts.index].values, 
            marker_color=['lightcoral' if label == st.session_state.selected_class else 'grey' for label in sorted_labels]
        )
    ])

    # update layout
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300, # reduce height
        barmode='group' # group bars
    )

    # display the figure
    st.plotly_chart(fig)
    logging.info("Class counts displayed as histogram successfully.")

# show results
def visualize():
    
    # display overall metrics
    with st.expander("**Overall Metrics**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{st.session_state.get('accuracy', 0):.2f}")
        col2.metric("Precision", f"{st.session_state.get('precision', 0):.2f}")
        col3.metric("Recall", f"{st.session_state.get('recall', 0):.2f}")
        col4.metric("F1 Score", f"{st.session_state.get('f1', 0):.2f}")
        logging.info("Overall metrics displayed successfully.")

    with st.expander(f"**Metrics for Class: {st.session_state.selected_class}**", expanded=True):
        class_y_test = (st.session_state.y_test == st.session_state.selected_class_index).astype(int)
        class_y_pred = (st.session_state.y_pred == st.session_state.selected_class_index).astype(int)
        class_accuracy = accuracy_score(class_y_test, class_y_pred)
        class_precision = precision_score(class_y_test, class_y_pred)
        class_recall = recall_score(class_y_test, class_y_pred)
        class_f1 = f1_score(class_y_test, class_y_pred)

        cm_error = False
        try:
            tn, fp, fn, tp = confusion_matrix(class_y_test, class_y_pred).ravel()
        except ValueError as e:
            cm_error = True
            tn, fp, fn, tp = 0, 0, 0, 0
            logging.error(f"Error in computing confusion matrix: {e}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{class_accuracy:.2f}")
        col2.metric("Precision", f"{class_precision:.2f}")
        col3.metric("Recall", f"{class_recall:.2f}")
        col4.metric("F1 Score", f"{class_f1:.2f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("True Positives", tp)
        col2.metric("True Negatives", tn)
        col3.metric("False Positives", fp)
        col4.metric("False Negatives", fn)

        if cm_error:
            st.error("Confusion matrix did not return enough values. Metrics may not be accurate.")

        logging.info(f"Metrics for class {st.session_state.selected_class} displayed successfully.")    

    with st.expander("**Class Counts**", expanded=True):
        display_class_counts(st.session_state.y_test, st.session_state.y_pred, st.session_state.unique_labels)

    with st.expander("**Confusion Matrix**", expanded=True):
        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.cm,
            x=[f'Predicted: {label}' for label in st.session_state.unique_labels],
            y=[f'Actual: {label}' for label in st.session_state.unique_labels],
            colorscale='Blues',
            showscale=True
        ))

        # confusion matrix: improve layout 
        fig.update_layout(
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            xaxis=dict(tickmode='array', tickvals=list(range(len(st.session_state.unique_labels))), ticktext=st.session_state.unique_labels),
            yaxis=dict(tickmode='array', tickvals=list(range(len(st.session_state.unique_labels))), ticktext=st.session_state.unique_labels),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        # confusion matrix: add a rectangle shape to create a border effect
        fig.add_shape(
            type="rect",
            x0=-0.5, x1=len(st.session_state.unique_labels) - 0.5,
            y0=-0.5, y1=len(st.session_state.unique_labels) - 0.5,
            line=dict(color="black", width=1),
            fillcolor='rgba(0,0,0,0)'
        )

        # confusion matrix: add rectangles to highlight the selected class (row and column)
        fig.add_shape(
            type="rect",
            x0=-0.5, x1=len(st.session_state.unique_labels) - 0.5,
            y0=st.session_state.selected_class_index - 0.5, y1=st.session_state.selected_class_index + 0.5,
            line=dict(color="red", width=1)
        )
        fig.add_shape(
            type="rect",
            x0=st.session_state.selected_class_index - 0.5, x1=st.session_state.selected_class_index + 0.5,
            y0=-0.5, y1=len(st.session_state.unique_labels) - 0.5,
            line=dict(color="red", width=1)
        )
        
        # confusion matrix: display the figure
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
            st.session_state.selected_class = st.selectbox("Select Class", st.session_state.unique_labels, index=st.session_state.selected_class_index)

        st.session_state.selected_class_index = list(st.session_state.unique_labels).index(st.session_state.selected_class)
        
        visualize()
        logging.info("Results displayed successfully.")
        # update selected class index when the selected class changes
