# VisA

Visual analytics app for exploring machine learning classifications.

[![Unit Tests](https://github.com/noelkronenberg/hub-visa/actions/workflows/tests.yml/badge.svg)](https://github.com/noelkronenberg/hub-visa/actions/workflows/tests.yml)

## Showcase

The app is live and hosted on the Streamlit Community Cloud: [visa-demo.streamlit.app](https://visa-demo.streamlit.app/)

## Structure

- ```.github/workflows/``` Directory containing GitHub Actions configurations.
-   - `tests.yml` Configuration for running unit tests on commit.
- ```.streamlit/``` Directory containing Streamlit configurations.
  - `config.toml` Configuration file for Streamlit server settings.
- ```app/``` Directory containing Streamlit application files.
  - `services/` Directory containing supporting files.
    - `data.py` Contains functions for loading and preparing data.
    - `error_analysis.py` Contains functions for visualizing error analysis.
    - `feature_importance.py` Contains functions for visualizing feature importance.
    - `model.py` Contains functions for training and evaluating the machine learning model.
  - `__init__.py` Initialization file for the app module.
  - `app.py` Main application file for the Streamlit dashboard.
  - `config.py` Configuration file for general settings used in the app.
  - `requirements.txt` Lists the Python packages required to run the app.
  - `lucas_organic_carbon/` Directory containing data files for the Lucas Organic Carbon dataset.
    - `target/` Directory containing target data files.
    - `training_test/` Directory containing training and test data files.
  - `test_app.py` Unit tests for checking the app.
- ```check_env.py``` Script to check if the required environment and packages are installed.
- ```environment.yml``` Conda environment configuration file listing the dependencies.
- ```local-install-instructions.md``` Instructions for setting up the project locally.

## Milestones

> Solved tasks and addressing of milestones.

| ID | Milestone | Solved Tasks | Improvements |
|---|------------|--------------|--------------|
| M1 | Data Exploration | Raw data attributes, overall data distributions, distribution of organic carbon concentration classes, spectral profiles of random soil samples, boxplot of selected wavelengths by carbon concentration class, profiling report. ||
| M2 | Random Forest                      | Label encoding, train-test split (test size = 0.2), grid search as well as randomized search with 3-fold CV on parameter grid (```n_estimators```, ```max_depth```, ```min_samples_split```, ```min_samples_leaf```, ```max_features```), evaluation of different grid search results (plot of score over iterations, confusion matrix, accuracy score, cross-validation score, mean cross-validation score). ||
| M3 | Explorative Error Analysis Concept (*Konzept I*) | [[presentation slides](https://docs.google.com/presentation/d/16qUn8gltr5sOPD6g4-4v1JwBQHDfBOIbRnPZEQq2H_U/edit?usp=sharing)] [[updated wireframe](https://docs.google.com/presentation/d/1CWkKQfMkITK6Dze0oZnR4_OqrqThp5agR9G6SoI4Jhk/edit?usp=sharing)] ||
| M4 | Explorative Error Analysis Prototype (*Komponente I*) | [see *Components*] ||
|M5|Feature Importance Concept (*Konzept II*)|[[presentation slides](https://docs.google.com/presentation/d/1sSXYiWVSzP-jKvBnRYrvyK_wPkAHrHsJ0h6WnJ6E7G0/edit?usp=sharing)]||
|M6|Feature Importance Prototype (*Komponente II*)|[see *Components*]|[see *Improvements*]|

## Components

> Major components of the VA system and the status of implementation.

| ID  | Component                     | Description                                                                                                     | Status | Milestone |
| --- | ----------------------------- | --------------------------------------------------------------------------------------------------------------- | ------ | --------- |
| C1  | Confusion Matrix              | Allow user to view confusion matrix for a trained model in Streamlit application.                               | done   | M4        |
| C2  | Evaluation Metrics            | Allow user to view specific model evaluation metrics.                                                           | done   | M4        |
| C3  | Data Upload                   | Allow user to upload their own or large dataset.                                                                | done   | M4        |
| C4  | Hosting                       | Allow user to view Streamlit application on a hosted website.                                                   | done   | M4        |
| C5  | Dynamic Model Training        | Allow user to change model parameters to retrain the model dynamically.                                         | done   | M4        |
| C6  | Faster UX                     | Enable faster loading times and improve usability.                                                              | done   | M4        |
| C7  | Class Selection               | Allow users to select a specific target class to view evaluation metrics.                                       | done   | M4        |
| C8  | Overview of Importance Scores | Show importance scores for each feature as bar chart.                                                           | done   | M6        |
| C9  | Impact of Intervals           | Show the average impact of intervals as beeswarm.                                                               | open   | M6        |
| C10 | Impact of 2-D Intervals       | Show the average impact of 2-D intervals as heatmap.                                                            | open   | M6        |
| C11 | Interval Settings             | Allow the user to change settings for intervals (e.g. number of intervals).                                     | open   | M6        |
| C12 | Improved Division of Tasks    | Allow the user to focus on a single task (e.g. model training, error exploration, investigation of importance). | done   | M6        |


## Activities

> Major implementation activities for each component.

| ID  | Component                     | Description                                                                 | Status | Point Person               |
| --- | ----------------------------- | --------------------------------------------------------------------------- | ------ | -------------------------- |
| C1  | Confusion Matrix              |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Setting up Streamlit application.                                           | done   | Noel Kronenberg            |
| A2  |                               | Integrating trained model with application.                                 | done   | Noel Kronenberg            |
| A3  |                               | Integrating trained model with Plotly figure.                               | done   | Noel Kronenberg            |
| A4  |                               | Integrating Plotly figure with application.                                 | done   | Noel Kronenberg            |
| C2  | Evaluation Metrics            |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Calculation of evaluation metrics for trained model.                        | done   | Noel Kronenberg            |
| A2  |                               | Displaying of evaluation metrics on Streamlit application.                  | done   | Noel Kronenberg            |
| A3  |                               | Adding a bar chart for the comparison of predicted and actual class counts. | done   | Aodi Chen                  |
| A4  |                               | Adding confusion matrix metrics.                                            | done   | Aodi Chen                  |
| A5  |                               | Adding collapsible sections to hide metrics.                                | done   | Noel Kronenberg            |
| A6  |                               | Preselecting class with lowest accuracy.                                    | done   | Noel Kronenberg            |
| C3  | Data Upload                   |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Adding form for uploading data.                                             | done   | Noel Kronenberg            |
| A2  |                               | Increasing maximum Streamlit upload limit.                                  | done   | Noel Kronenberg            |
| C4  | Hosting                       |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Setting up Streamlit Community Cloud.                                       | done   | Noel Kronenberg            |
| A2  |                               | Making application compatible with hosting.                                 | done   | Noel Kronenberg            |
| C5  | Dynamic Model Training        |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Adding form for adjustment of parameters.                                   | done   | Noel Kronenberg            |
| A2  |                               | Adding function to dynamically train model with new parameters.             | done   | Noel Kronenberg            |
| C6  | Faster UX                     |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Adding caching functionality of data (e.g. uploaded data).                  | done   | Noel Kronenberg            |
| A2  |                               | Adding caching functionality of resources (e.g. trained model).             | done   | Noel Kronenberg            |
| A3  |                               | Adding data loader signs for transparent loading processes.                 | done   | Noel Kronenberg            |
| C7  | Class Selection               |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Adding form for selection of class.                                         | done   | Noel Kronenberg            |
| A2  |                               | Highlighting class in confusion matrix.                                     | done   | Noel Kronenberg            |
| A3  |                               | Adding evaluation metrics for selected class.                               | done   | Noel Kronenberg            |
| A4  |                               | Adding highlight to evaluation metrics for selected class.                  | done   | Noel Kronenberg            |
| C8  | Overview of Importance Scores |                                                                             |        | Aodi Chen                  |
| A1  |                               | Calculating importance scores.                                              | done   | Aodi Chen, Noel Kronenberg |
| A2  |                               | Adding bar chart to plot feature importance.                                | done   | Aodi Chen, Noel Kronenberg |
| A3  |                               | Adding slider to select number of features.                                 | done   | Noel Kronenberg            |
| C9  | Impact of Intervals           |                                                                             |        | Fabian Henning             |
| C10 | Impact of 2-D Intervals       |                                                                             |        | Aodi Chen                  |
| C11 | Interval Settings             |                                                                             |        |                            |
| C12 | Improved Division of Tasks    |                                                                             |        | Noel Kronenberg            |
| A1  |                               | Adding of tabs for Explorative Error Analysis and Feature Importance.       | done   | Noel Kronenberg            |
| A2  |                               | Refactoring (e.g. encapsulating) the code to be more readable.              | done   | Noel Kronenberg            |

## Improvements

> Large optimizations to the components that are not core to official milestones tasks.

| ID | Improvement | Solved Tasks | Status | Point Person | Milestone |
|----|-------------|--------------|--------|--------------|-----------|
| I1 | Data Exploration Graphs | Bar chart for distribution of organic carbon concentration classes, spectral profiles of random soil samples, boxplot of selected wavelengths by carbon concentration class. | done | Noel Kronenberg | M6 |
| I2 | Model Download | Option to download trained model as a pickle file. | done | Noel Kronenberg | M6 |
| I3 | Demo Datasets | Option to choose from multiple demo datasets. | done | Noel Kronenberg | M6 |
| I4 | Unit Tests | Unit tests to check the app with automation for commits. | done | Noel Kronenberg | M6 |
