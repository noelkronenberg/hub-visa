import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import logging

# preset data
preset_target = 'app/lucas_organic_carbon/target/lucas_organic_carbon_target.csv'
preset_training = 'app/lucas_organic_carbon/training_test/compressed_data.csv'

# load data
@st.cache_data(show_spinner=False)
def load_data(load_preset_target, load_preset_training, target_data=None, training_data=None):
    
    # load data if not in session state
    if load_preset_target:
        df_target = pd.read_csv(preset_target)
        logging.info("Loaded target data from preset.")
    else:
        df_target = target_data
        logging.info("Loaded target data from uploaded file.")

    if load_preset_training:
        df_training = pd.read_csv(preset_training) # NOTE: this is a compressed version of the data (for demo purposes)
        logging.info("Loaded training data from preset.")
    else:
        df_training = training_data
        logging.info("Loaded training data from uploaded file.")

    df_combined = pd.merge(df_training, df_target, left_index=True, right_index=True)

    return df_training, df_target, df_combined

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

def plot_target_distribution(df_combined):
    """
    Plot target distribution.
    """

    target_column = df_combined.columns[-1]

    # create histogram
    fig = px.histogram(df_combined, x=target_column, color=target_column, title='', 
                       labels={target_column: 'Target Class'}, 
                       template='plotly_dark',
                       category_orders={target_column: df_combined[target_column].value_counts().index})
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), yaxis_title='Count')

    # update hover template
    fig.update_traces(hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>')

    # display the figure
    with st.expander("**Target Distribution**", expanded=True):
        st.plotly_chart(fig, key='target_distribution')
        logging.info("Target distribution displayed successfully.")

def plot_feature_profiles(df_combined):
    """
    Plot feature profiles.
    """

    target_column = df_combined.columns[-1]

    with st.expander("**Feature Profiles**", expanded=True):
        # sample feature profiles
        num_samples = st.slider('Number of Samples', min_value=1, max_value=10, value=3, key='num_samples_feature')

        # select a few feature profiles to display
        sample_feature_profiles = df_combined.sample(num_samples).drop(columns=[target_column])

        # plot feature profiles
        fig_feature = px.line(title='')
        for idx, row in sample_feature_profiles.iterrows():
            fig_feature.add_scatter(x=row.index.astype(float), y=row.values, mode='lines', name=f'Sample {idx}', hovertemplate='Feature: %{x}<br>Value: %{y}<extra></extra>')
        fig_feature.update_layout(xaxis_title='Feature', yaxis_title='Value', template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_feature, key='feature_profiles')

        logging.info("Feature profiles displayed successfully.")

def plot_feature_distribution(df_combined):
    """
    Plot feature distribution.
    """

    target_column = df_combined.columns[-1]

    # select a few features to display (first, 1/4, 1/2, 3/4, last)
    feature_columns = df_combined.columns[:-1]
    selected_features = [
        feature_columns[0],
        feature_columns[len(feature_columns) // 4],
        feature_columns[len(feature_columns) // 2],
        feature_columns[3 * len(feature_columns) // 4],
        feature_columns[-1]
    ]

    # melt the DataFrame for plotting (wide to long format)
    df_melted = df_combined.melt(id_vars=target_column, value_vars=selected_features, var_name='Feature', value_name='Value')

    # create box plot
    fig_boxplot = px.box(df_melted, x='Feature', y='Value', color=target_column, title='', template='plotly_dark')
    fig_boxplot.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    # update hover template
    fig_boxplot.update_traces(hovertemplate='Feature: %{x}<br>Value: %{y}<extra></extra>')

    # display the figure
    with st.expander("**Feature Distribution**", expanded=True):
        st.plotly_chart(fig_boxplot, key='feature_distribution')
        logging.info("Feature distribution displayed successfully.")