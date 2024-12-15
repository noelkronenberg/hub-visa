# Development and Implementation Documentation

## Structure of Repository

- ```.streamlit``` Streamlit configuration (e.g. server settings) 
- ```app``` Streamlit-based application
- ```data-exploration``` exploratory data analysis of raw data
- ```model-training``` training of different models

## Purpose and Scope

> Solved tasks and addressing of milestones.

| ID | Milestone                          | Solved Tasks                                                                                       |
|--------------|------------------------------------|---------------------------------------------------------------------------------------------------|
| M1 | Data Exploration | Raw data attributes, overall data distributions, distribution of organic carbon concentration classes, spectral profiles of random soil samples, boxplot of selected wavelengths by carbon concentration class, profiling report. |
| M2 | Random Forest                      | Train-test split (test size = 0.2), grid search as well as randomized search with 3-fold CV on parameter grid (```n_estimators```, ```max_depth```, ```min_samples_split```, ```min_samples_leaf```, ```max_features```), evaluation of different grid search results (plot of score over iterations, confusion matrix, accuracy score, cross-validation score, mean cross-validation score). |
| M3 | Explorative Error Analysis Concept | [see [presentation slides](https://docs.google.com/presentation/d/16qUn8gltr5sOPD6g4-4v1JwBQHDfBOIbRnPZEQq2H_U/edit?usp=sharing)] |
| M4 | Explorative Error Analysis Prototype | [see *Components*] |

## Components

> Major components of the VA system and the status of implementation.

| ID  | Component                | Description                                                                               | Status            | Milestone |
| --- | ------------------------ | ----------------------------------------------------------------------------------------- | ----------------- | --------- |
| C1  | Confusion Matrix         | Allow user to view confusion matrix for a trained model in Streamlit application.         | done              | M4        |
| C2  | Evaluation Metrics       | Allow user to view specific evaluation metrics of model.                                  | done              | M4        |
| C3  | Data Upload              | Allow user to upload their own or large dataset.                                          | done              | M4        |
| C4  | Hosting                  | Allow user to view Streamlit application on hosted website.                               | done              | M4        |
| C5  | Dynamic Model Training   | Allow user to change model parameters to dynamically retrain model.                       | done              | M4        |
| C6  | Faster UX                | Enabling faster loading times via caching of resources and transparent loading processes. | done              | M4        |
| C7  | Class Selection          | Allow user to select specific target class to view evaluation metrics regarding it.       | done              | M4        |
| C8  | Improved Metrics Display | Improve readability of key evaluation metrics.                                            | under development | M4        |


## Major Implementation Activities

> Major implementation activities for each component.

| ID  | Component                | Description                                                                  | Status | Point Person    |
| --- | ------------------------ | ---------------------------------------------------------------------------- | ------ | --------------- |
| C1  | Confusion Matrix         |                                                                              |        |                 |
| A1  |                          | Setting up Streamlit application.                                            | done   | Noel Kronenberg |
| A2  |                          | Integrating trained model with application.                                  | done   | Noel Kronenberg |
| A3  |                          | Integrating trained model with Plotly figure.                                | done   | Noel Kronenberg |
| A4  |                          | Integrating Plotly figure with application.                                  | done   | Noel Kronenberg |
| C2  | Evaluation Metrics       |                                                                              |        |                 |
| A1  |                          | Calculation of evaluation metrics for trained model.                         | done   | Noel Kronenberg |
| A2  |                          | Displaying of evaluation metrics on Streamlit application.                   | done   | Noel Kronenberg |
| C3  | Data Upload              |                                                                              |        |                 |
| A1  |                          | Adding form for uploading data.                                              | done   | Noel Kronenberg |
| A2  |                          | Increasing maximum Streamlit upload limit.                                   | done   | Noel Kronenberg |
| C4  | Hosting                  |                                                                              |        |                 |
| A1  |                          | Setting up Streamlit Community Cloud.                                        | done   | Noel Kronenberg |
| A2  |                          | Making application compatible with hosting.                                  | done   | Noel Kronenberg |
| C5  | Dynamic Model Training   |                                                                              |        |                 |
| A1  |                          | Adding form for adjustment of parameters.                                    | done   | Noel Kronenberg |
| A2  |                          | Adding function to dynamically train model with new parameters.              | done   | Noel Kronenberg |
| C6  | Faster UX                |                                                                              |        |                 |
| A1  |                          | Adding caching functionality of data (e.g. uploaded data).                   | done   | Noel Kronenberg |
| A2  |                          | Adding caching functionality of resources (e.g. trained model).              | done   | Noel Kronenberg |
| A3  |                          | Adding of data loader signs for transparent loading processes.               | done   | Noel Kronenberg |
| C7  | Class Selection          |                                                                              |        |                 |
| A1  |                          | Adding form for selection of class.                                          | done   | Noel Kronenberg |
| A2  |                          | Highlighting class in confusion matrix.                                      | done   | Noel Kronenberg |
| A3  |                          | Adding evaluation metrics for selected class.                                | done   | Noel Kronenberg |
| A4  |                          | Adding highlight to evaluation metrics for selected class.                   | done   | Noel Kronenberg |
| C8  | Improved Metrics Display |                                                                              |        |                 |
| A1  |                          | Adding a bar chart for the comparison for predicted and actual class counts. | done   | Aodi Chen       |
| A2  |                          | Adding confusion matrix metrics.                                             | done   | Aodi Chen       |
| A3  |                          | Adding collapsible sections.                                                 | done   | Noel Kronenberg |
| A4  |                          | Preselect class with lowest accuracy.                                        | done   | Noel Kronenberg |