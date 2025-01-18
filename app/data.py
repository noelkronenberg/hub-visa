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

    # create histogram
    fig = px.histogram(df_combined, x='x', color='x', title='', 
                       labels={'x': 'Carbon Concentration Class'}, 
                       template='plotly_dark',
                       category_orders={'x': df_combined['x'].value_counts().index})
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    # update hover template
    fig.update_traces(hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>')

    # display the figure
    with st.expander("**Target Distribution**", expanded=True):
        st.plotly_chart(fig)
        logging.info("Target distribution displayed successfully.")

def plot_spectral_profiles(df_combined):
    """
    Plot spectral profiles.
    """

    with st.expander("**Spectral Profiles**", expanded=True):
        # sample spectral profiles
        num_samples = st.slider('Number of Samples', min_value=1, max_value=10, value=3, key='num_samples_spectral')

        # select a few spectral profiles to display
        sample_spectral_profiles = df_combined.sample(num_samples).drop(columns=['x'])

        # plot spectral profiles
        fig_spectral = px.line(title='')
        for idx, row in sample_spectral_profiles.iterrows():
            fig_spectral.add_scatter(x=row.index.astype(float), y=row.values, mode='lines', name=f'Sample {idx}', hovertemplate='Wavelength: %{x}<br>Spectral Value: %{y}<extra></extra>')
        fig_spectral.update_layout(xaxis_title='Wavelength', yaxis_title='Spectral Value', template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_spectral)

        logging.info("Spectral profiles displayed successfully.")

def plot_wavelength_distribution(df_combined):
    """
    Plot wavelength distribution.
    """

    # select a few wavelengths to display (first, 1/4, 1/2, 3/4, last)
    wavelength_columns = df_combined.columns[:-1]
    selected_wavelengths = [
        wavelength_columns[0],
        wavelength_columns[len(wavelength_columns) // 4],
        wavelength_columns[len(wavelength_columns) // 2],
        wavelength_columns[3 * len(wavelength_columns) // 4],
        wavelength_columns[-1]
    ]

    # melt the DataFrame for plotting (wide to long format)
    df_melted = df_combined.melt(id_vars='x', value_vars=selected_wavelengths, var_name='Wavelength', value_name='Value')

    # create box plot
    fig_boxplot = px.box(df_melted, x='Wavelength', y='Value', color='x', title='', template='plotly_dark')
    fig_boxplot.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    # update hover template
    fig_boxplot.update_traces(hovertemplate='Wavelength: %{x}<br>Value: %{y}<extra></extra>')

    # display the figure
    with st.expander("**Wavelength Distribution**", expanded=True):
        st.plotly_chart(fig_boxplot)
        logging.info("Wavelength distribution displayed successfully.")