import logging
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go

from config import SELECTION_COLOR

def _display_overall_metrics(accuracy, precision, recall, f1):
    """
    Display the overall metrics.
    """

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")
    logging.info("Overall metrics displayed successfully.")

def _display_class_counts(y_test, y_pred, unique_labels, selected_class):
    """
    Display the class counts as a histogram.
    """

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
            marker_color='black',
            marker_line_width=1,
            marker_line_color=[SELECTION_COLOR if label == selected_class else 'black' for label in sorted_labels],
            hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>'
        ),
        go.Bar(
            name='Predicted',
            x=sorted_labels, 
            y=predicted_class_counts[sorted_counts.index].values, 
            marker_color='grey',
            marker_line_color=[SELECTION_COLOR if label == selected_class else 'grey' for label in sorted_labels],
            marker_line_width=1,
            hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>'
        )
    ])

    # update layout
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300, # reduce height
        barmode='group', # group bars
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(sorted_labels))),
            ticktext=[f'<span style="color:{SELECTION_COLOR};">{label}</span>' if label == selected_class else label for label in sorted_labels]
        )
    )

    # display the figure
    st.plotly_chart(fig)
    logging.info("Class counts displayed as histogram successfully.")

def _display_class_metrics(y_test, y_pred, selected_class, selected_class_index):
    """
    Display the metrics for the selected class.
    """

    class_y_test = (y_test == selected_class_index).astype(int)
    class_y_pred = (y_pred == selected_class_index).astype(int)
    class_accuracy = accuracy_score(class_y_test, class_y_pred)
    class_precision = precision_score(class_y_test, class_y_pred)
    class_recall = recall_score(class_y_test, class_y_pred)
    class_f1 = f1_score(class_y_test, class_y_pred)

    # compute confusion matrix
    cm_error = False
    try:
        tn, fp, fn, tp = confusion_matrix(class_y_test, class_y_pred).ravel()
    except ValueError as e:
        cm_error = True
        tn, fp, fn, tp = 0, 0, 0, 0
        logging.error(f"Error in computing confusion matrix: {e}")

    # display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{class_accuracy:.2f}")
    col2.metric("Precision", f"{class_precision:.2f}")
    col3.metric("Recall", f"{class_recall:.2f}")
    col4.metric("F1 Score", f"{class_f1:.2f}")

    # display confusion matrix metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Positives", tp)
    col2.metric("True Negatives", tn)
    col3.metric("False Positives", fp)
    col4.metric("False Negatives", fn)

    if cm_error:
        st.error("Confusion matrix did not return enough values. Metrics may not be accurate.")

    logging.info(f"Metrics for class {selected_class} displayed successfully.")    

def _display_confusion_matrix(cm, unique_labels, selected_class_index):
    """
    Display the confusion matrix.
    """

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'{label}' for label in unique_labels],
        y=[f'{label}' for label in unique_labels],
        colorscale='Blues',
        showscale=True,
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))

    # confusion matrix: improve layout 
    fig.update_layout(
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        xaxis=dict(
            tickmode='array', 
            tickvals=list(range(len(unique_labels))), 
            ticktext=[f'<span style="color:{SELECTION_COLOR};">{label}</span>' if i == selected_class_index else label for i, label in enumerate(unique_labels)]
        ),
        yaxis=dict(
            tickmode='array', 
            tickvals=list(range(len(unique_labels))), 
            ticktext=[f'<span style="color:{SELECTION_COLOR};">{label}</span>' if i == selected_class_index else label for i, label in enumerate(unique_labels)]
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # add a rectangle shape to create a border effect
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=len(unique_labels) - 0.5,
        y0=-0.5, y1=len(unique_labels) - 0.5,
        line=dict(color="black", width=1),
        fillcolor='rgba(0,0,0,0)'
    )

    # add rectangles to highlight the selected class (row and column)
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=len(unique_labels) - 0.5,
        y0=selected_class_index - 0.5, y1=selected_class_index + 0.5,
        line=dict(color=SELECTION_COLOR, width=1),
    )
    fig.add_shape(
        type="rect",
        x0=selected_class_index - 0.5, x1=selected_class_index + 0.5,
        y0=-0.5, y1=len(unique_labels) - 0.5,
        line=dict(color=SELECTION_COLOR, width=1),
    )
    
    # display the figure
    st.plotly_chart(fig)
    logging.info("Confusion matrix displayed successfully.")

# show results
def visualize_error_analysis(y_test, y_pred, unique_labels, selected_class, selected_class_index, accuracy, precision, recall, f1, cm):
    """
    Display the error analysis results.
    """

    # display overall metrics
    with st.expander("**Overall Metrics**", expanded=True):
        _display_overall_metrics(accuracy, precision, recall, f1)

    # display metrics for the selected class
    with st.expander(f"**Metrics for Class: {selected_class}**", expanded=True):
       _display_class_metrics(y_test, y_pred, selected_class, selected_class_index)

    with st.expander("**Class Counts**", expanded=True):
        _display_class_counts(y_test, y_pred, unique_labels, selected_class)

    with st.expander("**Confusion Matrix**", expanded=True):
        _display_confusion_matrix(cm, unique_labels, selected_class_index)