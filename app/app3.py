import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import logging
import joblib
import os

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# streamlit app
st.set_page_config(page_title="VisA Dashboard", layout="wide")
st.title("VisA Dashboard")

# initialize session state
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 14
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 384
if 'data_percentage' not in st.session_state:
    st.session_state.data_percentage = 1
if 'normalize_cm' not in st.session_state:
    st.session_state.normalize_cm = True
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

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
        X_train, X_test, y_train, y_test, label_encoder = prepare_data(df_combined, data_percentage)
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['label_encoder'] = label_encoder
        logging.info("Data prepared successfully.")
    
    # spinner while training model
    with st.spinner('Training model...'):
        rf_classifier = train_model(X_train, y_train, max_depth, n_estimators)
        logging.info("Model trained successfully.")
        
        # Make predictions
        y_pred = rf_classifier.predict(X_test)
        st.session_state['y_pred'] = y_pred

    # spinner while evaluating
    with st.spinner('Evaluating the model...'):
        # compute metrics (and save in session state)
        st.session_state.accuracy = accuracy_score(y_test, y_pred)
        st.session_state.precision = precision_score(y_test, y_pred, average='weighted')
        st.session_state.recall = recall_score(y_test, y_pred, average='weighted')
        st.session_state.f1 = f1_score(y_test, y_pred, average='weighted')

        # actual label names (and save in session state)
        st.session_state.unique_labels = label_encoder.classes_

        # compute confusion matrix (and save in session state)
        if normalize_cm:
            st.session_state.cm = confusion_matrix(y_test, y_pred, labels=range(len(st.session_state.unique_labels)), normalize='true')
        else:
            st.session_state.cm = confusion_matrix(y_test, y_pred, labels=range(len(st.session_state.unique_labels)))

        logging.info("Model evaluated successfully.")
    
    st.session_state.first_run = False

# show results
def visualize():
    required_keys = ['label_encoder', 'X_test', 'y_test', 'y_pred', 'accuracy', 'precision', 'recall', 'f1', 'unique_labels', 'cm']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.warning("Please click 'Update' to train the model first.")
        return
    
    label_encoder = st.session_state['label_encoder']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']
    
    st.subheader("Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{st.session_state.accuracy:.2f}")
    col2.metric("Precision", f"{st.session_state.precision:.2f}")
    col3.metric("Recall", f"{st.session_state.recall:.2f}")
    col4.metric("F1 Score", f"{st.session_state.f1:.2f}")
    
    # Task 1: Confusion Matrix
    st.subheader('Task1-Identify Overall Misclassification Patterns')
    st.markdown("""
    **Task Description**: 
   Users want to understand which lasses are frequently confused with others, providing insights into systematic misclassification trends
    """)

    hover_text = []
    for i in range(len(st.session_state.unique_labels)):
        hover_text.append([])
        for j in range(len(st.session_state.unique_labels)):
            value = st.session_state.cm[i][j]
            
            # count TP, FP, FN, TN
            if i == j: 
                tp_tn = "TP" if i == 1 else "TN"
                text = f"{tp_tn}: {value:.2f}<br>"
            else:  
                fp_fn = "FP" if i == 0 else "FN"
                text = f"{fp_fn}: {value:.2f}<br>"
                
            text += f"Actual: {st.session_state.unique_labels[i]}<br>"
            text += f"Predicted: {st.session_state.unique_labels[j]}<br>"
            text += f"Value: {value:.2f}"
            
            hover_text[i].append(text)

    fig = go.Figure(data=go.Heatmap(
        z=st.session_state.cm,
        x=[f'Predicted: {label}' for label in st.session_state.unique_labels],
        y=[f'Actual: {label}' for label in st.session_state.unique_labels],
        colorscale=[
        [0, '#deebf7'],    
        [0.5, '#9ecae1'],  
        [1, '#3182bd']     
    ],
        showscale=True,
        hoverongaps=False,
        hoverinfo='text',
        text=hover_text,
        hovertemplate="%{text}<extra></extra>"  
    ))
    
    fig.update_layout(
        
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        xaxis=dict(tickmode='array', tickvals=list(range(len(st.session_state.unique_labels))), ticktext=st.session_state.unique_labels),
        yaxis=dict(tickmode='array', tickvals=list(range(len(st.session_state.unique_labels))), ticktext=st.session_state.unique_labels)
    )
    left_col, center_col, right_col = st.columns([1,2,1])
    with center_col:
        st.plotly_chart(fig, use_container_width=True)
    logging.info("Confusion matrix displayed successfully.")

    # Task 2: Prediction Comparison
    st.subheader('Task2-Compare True Labels with Predicted Labels to Identify Major Misclassifications')
    st.markdown("""
    **Task Description**: 
    For each data point, users want to compare the true class with the predicted classes to quickly pinpoint which 
    misclassifications are most significant and require attention.
    """)


    true_labels = label_encoder.inverse_transform(y_test)
    pred_labels = label_encoder.inverse_transform(y_pred)
    unique_classes = np.unique(true_labels)

    raw_cm = confusion_matrix(y_test, y_pred, labels=range(len(st.session_state.unique_labels)))
    
    col_sums = raw_cm.sum(axis=0)  
    col_correct = raw_cm.diagonal()  #
    col_accuracy = np.round(col_correct / col_sums * 100, 2) 
    
    comparison_fig = go.Figure(data=go.Heatmap(
        z=raw_cm,
        x=[f'Pred: {label}<br>Acc: {acc}%' for label, acc in zip(st.session_state.unique_labels, col_accuracy)],
        y=[f'True: {label}' for label in st.session_state.unique_labels],
        colorscale=[
            [0, '#f7fbff'],    
            [0.2, '#deebf7'],  
            [0.4, '#c6dbef'], 
            [0.6, '#9ecae1'],  
            [0.8, '#6baed6'], 
            [1, '#2171b5']     
        ],
        showscale=True,
        text=raw_cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<br>Accuracy: %{customdata}%<extra></extra>",
        customdata=[[acc for acc in col_accuracy] for _ in range(len(st.session_state.unique_labels))]
    ))

    comparison_fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(
            autorange="reversed",
            tickmode='array',
            ticktext=st.session_state.unique_labels,
            tickvals=list(range(len(st.session_state.unique_labels)))
        ),
        xaxis=dict(
            tickmode='array',
            ticktext=[f'{label}<br>Acc: {acc}%' for label, acc in zip(st.session_state.unique_labels, col_accuracy)],
            tickvals=list(range(len(st.session_state.unique_labels)))
        )
    )

    left_col, center_col, right_col = st.columns([1,2,1])
    with center_col:
        st.plotly_chart(comparison_fig, use_container_width=True)

   

    # Task 3: Sankey Diagram
    st.subheader('Task3-Detect Class Imbalance Impact')
    st.markdown("""
    **Task Description**: 
    Users want to identify how class imbalances affect performance and whether certain classes are under- or overrepresented in predictions.
    """)

    true_labels = label_encoder.inverse_transform(y_test)
    pred_labels = label_encoder.inverse_transform(y_pred)
    
    unique_classes = np.unique(true_labels)
    
    source_labels = [f"True_{label}" for label in unique_classes]
    target_labels = [f"Pred_{label}" for label in unique_classes]
    

    node_labels = source_labels + target_labels
    
    flows = []
    sources = []
    targets = []
    
    for i, true_class in enumerate(unique_classes):
        for j, pred_class in enumerate(unique_classes):
            flow = sum((true_labels == true_class) & (pred_labels == pred_class))
            if flow > 0:  
                flows.append(flow)
                sources.append(i)
                targets.append(j + len(unique_classes))
    
    sankey_fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = node_labels,
            color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] * 2 
        ),
        link = dict(
            source = sources,
            target = targets,
            value = flows,
            color = ['rgba(31, 119, 180, 0.4)'] * len(flows) 
        )
    )])
    
 
    sankey_fig.update_layout(
        title_text="Flow of Predictions between True and Predicted Classes",
        font_size=10,
        height=500
    )
    

    left_col, center_col, right_col = st.columns([1,2,1])
    with center_col:
        st.plotly_chart(sankey_fig, use_container_width=True)
        
    st.subheader("Class Distribution")
    
    true_dist = pd.Series(true_labels).value_counts()
    pred_dist = pd.Series(pred_labels).value_counts()
    
    
    dist_fig = go.Figure()
    
    dist_fig.add_trace(go.Bar(
        x=true_dist.index,
        y=true_dist.values,
        name='True Distribution',
        marker_color='rgba(31, 119, 180, 0.7)'
    ))
    
    dist_fig.add_trace(go.Bar(
        x=pred_dist.index,
        y=pred_dist.values,
        name='Predicted Distribution',
        marker_color='rgba(255, 127, 14, 0.7)'
    ))
    
    dist_fig.update_layout(
        title='True vs Predicted Class Distribution',
        xaxis_title='Class',
        yaxis_title='Count',
        barmode='group',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    left_col, center_col, right_col = st.columns([1,2,1])
    with center_col:
        st.plotly_chart(dist_fig, use_container_width=True)

    # Task 4: Interactive Error Analysis
    st.subheader('Task4-Explore classification error for specification groups of data')
    st.markdown("""
    **Task Description**: 
    Users want to focus on subsets of data, such as particular classes or feature values, to explore classification error trends.
    """)
    
    true_labels = label_encoder.inverse_transform(y_test)
    pred_labels = label_encoder.inverse_transform(y_pred)
    unique_classes = np.unique(true_labels)

    selected_class = st.selectbox(
        "Select Class to Focus",
        ["All Classes"] + list(unique_classes),
        key="class_selector"
    )

  
    mask = np.ones(len(y_test), dtype=bool)
    if selected_class != "All Classes":
        mask = mask & (true_labels == selected_class)

    filtered_cm = confusion_matrix(
        y_test[mask], 
        y_pred[mask], 
        labels=range(len(unique_classes)),
        normalize='true' if normalize_cm else None
    )

    dynamic_fig = go.Figure(data=go.Heatmap(
        z=filtered_cm,
        x=[f'Predicted: {label}' for label in unique_classes],
        y=[f'Actual: {label}' for label in unique_classes],
        colorscale=[
            [0, '#deebf7'],   
            [0.5, '#9ecae1'],  
            [1, '#3182bd']     
        ],
        showscale=True,
        hoverinfo='text',
        text=[[f'True: {unique_classes[i]}<br>Predicted: {unique_classes[j]}<br>Value: {filtered_cm[i][j]:.2f}'
               for j in range(len(unique_classes))]
              for i in range(len(unique_classes))],
        hovertemplate="%{text}<extra></extra>"
    ))


    if selected_class != "All Classes":
        class_idx = list(unique_classes).index(selected_class)
        dynamic_fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=len(unique_classes)-0.5,
            y0=class_idx-0.5,
            y1=class_idx+0.5,
            line=dict(color="red", width=2),
            fillcolor="rgba(0,0,0,0)"
        )
        dynamic_fig.add_shape(
            type="rect",
            x0=class_idx-0.5,
            x1=class_idx+0.5,
            y0=-0.5,
            y1=len(unique_classes)-0.5,
            line=dict(color="blue", width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    dynamic_fig.update_layout(
        title=f"Interactive Confusion Matrix" + 
              (f" - Focusing on {selected_class}" if selected_class != "All Classes" else ""),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        xaxis=dict(tickmode='array', tickvals=list(range(len(unique_classes))), ticktext=unique_classes),
        yaxis=dict(tickmode='array', tickvals=list(range(len(unique_classes))), ticktext=unique_classes)
    )

    st.write(f"Number of samples in current selection: {sum(mask)}")

    left_col, center_col, right_col = st.columns([1,2,1])
    with center_col:
        st.plotly_chart(dynamic_fig, use_container_width=True)

    if selected_class != "All Classes":
        st.subheader(f"Detailed Error Analysis for {selected_class}")

        class_idx = list(unique_classes).index(selected_class)
        
        raw_filtered_cm = confusion_matrix(
            y_test[mask], 
            y_pred[mask], 
            labels=range(len(unique_classes)),
            normalize=None  
        )
        
        error_rates = pd.DataFrame({
            'Predicted Class': unique_classes,
            'Sample Count': raw_filtered_cm[class_idx],  
            'Error Rate (%)': filtered_cm[class_idx] / filtered_cm[class_idx].sum() * 100  
        })

        error_rates = error_rates[error_rates['Predicted Class'] != selected_class]
        error_rates = error_rates.sort_values('Error Rate (%)', ascending=False)

        st.write("Main misclassification patterns:")
        st.dataframe(
            error_rates.style.format({
                'Sample Count': '{:,.0f}',
                'Error Rate (%)': '{:.2f}%'
            }),
            hide_index=True  
        )
if __name__ == "__main__":
    if 'first_run' not in st.session_state or st.session_state.first_run:
        st.warning("Please adjust the parameters and click 'Update' to train the model.")
    else:
        visualize()