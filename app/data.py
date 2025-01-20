import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import logging
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# demo cases
demo_cases = {
    'Lucas Organic Carbon': {
        'target': 'app/lucas_organic_carbon/target/lucas_organic_carbon_target.csv',
        'training': 'app/lucas_organic_carbon/training_test/compressed_data.csv'
    },
    'Iris Dataset': {
        'sklearn_dataset': load_iris
    },
    'Wine Dataset': {
        'sklearn_dataset': load_wine
    },
    'Breast Cancer Dataset': {
        'sklearn_dataset': load_breast_cancer
    },
    'Custom Data': {
        'custom': True
    }
}

preset_target = demo_cases['Lucas Organic Carbon']['target']
preset_training = demo_cases['Lucas Organic Carbon']['training']

# load data
@st.cache_data(show_spinner=False)
def load_data(load_preset_target, load_preset_training, target_data=None, training_data=None, selected_demo_case='Lucas Organic Carbon'):
    """
    Load the data for training.
    
    Parameters:
    - load_preset_target: Deprecated. Use selected_demo_case instead. 
    - load_preset_training: Deprecated. Use selected_demo_case instead.
    """

    # TODO: remove load_preset_target and load_preset_training in future versions
    
    # load data from sklearn
    if 'sklearn_dataset' in demo_cases[selected_demo_case]:
        logging.info(f"Loading {selected_demo_case} data from sklearn.")
        data = demo_cases[selected_demo_case]['sklearn_dataset']()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df_combined = df
        df_training = df.drop(columns=['target'])
        df_target = df[['target']]
        logging.info(f"Loaded {selected_demo_case} data from sklearn.")
        logging.info(f"Training data shape: {df_training.shape}")
        logging.info(f"Target data shape: {df_target.shape}")
        logging.info(f"Combined data shape: {df_combined.shape}")

    # load custom data
    elif 'custom' in demo_cases[selected_demo_case]:
        logging.info("Loading custom data.")
        if st.session_state['custom_training'] and st.session_state['custom_target']:
            df_target = target_data
            df_training = training_data
            logging.info("Loaded custom target and training data from uploaded files.")
        else:
            logging.error("Full custom data not provided. Please upload both target and training data.")
            return None, None, None

        df_combined = pd.merge(df_training, df_target, left_index=True, right_index=True)

    # load data from presets
    else:
        df_target = pd.read_csv(demo_cases[selected_demo_case]['target'])
        logging.info("Loaded target data from preset.")

        df_training = pd.read_csv(demo_cases[selected_demo_case]['training'])
        logging.info("Loaded training data from preset.")
        
        df_combined = pd.merge(df_training, df_target, left_index=True, right_index=True)
        logging.info("Merged target and training data successfully.")

    return df_training, df_target, df_combined

# prepare data
@st.cache_data(show_spinner=False)
def prepare_data(df_combined, data_percentage):
    """
    Prepare the data for training.
    """

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

    # split the data with error handling
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data prepared successfully.")
    except ValueError:
        logging.error(f"Data preparation failed. Percentage of data used: {data_percentage}.")
        # st.warning("Data preparation failed. Please adjust percentage of data used.")
        return None, None, None, None, None

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
            fig_feature.add_scatter(x=row.index.astype(str), y=row.values, mode='lines', name=f'Sample {idx}', hovertemplate='Feature: %{x}<br>Value: %{y}<extra></extra>')
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
    fig_boxplot = px.box(df_melted, x='Feature', y='Value', color=target_column, title='', template='plotly_dark',
                         labels={target_column: 'Target Class'})
    fig_boxplot.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    # update hover template
    fig_boxplot.update_traces(hovertemplate='Feature: %{x}<br>Value: %{y}<extra></extra>')

    # display the figure
    with st.expander("**Feature Distribution**", expanded=True):
        st.plotly_chart(fig_boxplot, key='feature_distribution')
        logging.info("Feature distribution displayed successfully.")