# VisA

Visual analytics app for exploring machine learning classifications.

## Showcase

https://github.com/user-attachments/assets/27bd339a-1999-416d-a43f-ba3ad347cadf

> Video updated on December 15th 2024 (commit ```1fe9a5b```). In case the video is not accessible, you may view it via [this](https://drive.google.com/drive/folders/132BcJ8Trho9_eRaCsOj504YdOCG96X35?usp=sharing) link.

## Structure

- ```.streamlit``` Streamlit configuration (e.g. server settings) 
- ```app``` Streamlit-based application
- ```data-exploration``` exploratory data analysis of raw data
- ```model-training``` training of different models

## Milestones

> Solved tasks and addressing of milestones.

| ID | Milestone                          | Solved Tasks                                                                                       |
|--------------|------------------------------------|---------------------------------------------------------------------------------------------------|
| M1 | Data Exploration | Raw data attributes, overall data distributions, distribution of organic carbon concentration classes, spectral profiles of random soil samples, boxplot of selected wavelengths by carbon concentration class, profiling report. |
| M2 | Random Forest                      | Label encoding, train-test split (test size = 0.2), grid search as well as randomized search with 3-fold CV on parameter grid (```n_estimators```, ```max_depth```, ```min_samples_split```, ```min_samples_leaf```, ```max_features```), evaluation of different grid search results (plot of score over iterations, confusion matrix, accuracy score, cross-validation score, mean cross-validation score). |
| M3 | Explorative Error Analysis Concept (*Konzept I*) | [[presentation slides](https://docs.google.com/presentation/d/16qUn8gltr5sOPD6g4-4v1JwBQHDfBOIbRnPZEQq2H_U/edit?usp=sharing)] [[updated wireframe](https://docs.google.com/presentation/d/1CWkKQfMkITK6Dze0oZnR4_OqrqThp5agR9G6SoI4Jhk/edit?usp=sharing)] |
| M4 | Explorative Error Analysis Prototype (*Komponente I*) | [see *Components*] |
|M5|Feature Importance Concept (*Konzept II*)|[[presentation slides](https://docs.google.com/presentation/d/1sSXYiWVSzP-jKvBnRYrvyK_wPkAHrHsJ0h6WnJ6E7G0/edit?usp=sharing)]|
|M6|Feature Importance Prototype (*Komponente II*)|[see *Components*]|

## Components

> Major components of the VA system and the status of implementation.

| ID  | Component                | Description                                                                               | Status            | Milestone |
| --- | ------------------------ | ----------------------------------------------------------------------------------------- | ----------------- | --------- |
| C1  | Confusion Matrix         | Allow user to view confusion matrix for a trained model in Streamlit application.         | done              | M4        |
| C2  | Evaluation Metrics       | Allow user to view specific model evaluation metrics.                                     | done              | M4        |
| C3  | Data Upload              | Allow user to upload their own or large dataset.                                          | done              | M4        |
| C4  | Hosting                  | Allow user to view Streamlit application on a hosted website.                             | done              | M4        |
| C5  | Dynamic Model Training   | Allow user to change model parameters to retrain the model dynamically.                   | done              | M4        |
| C6  | Faster UX                | Enable faster loading times and improve usability.                                        | done              | M4        |
| C7  | Class Selection          | Allow users to select a specific target class to view evaluation metrics.                 | done              | M4        |
| C8  | Overview of Importance Scores | Show importance scores for each feature as bar chart.                                            | open              | M6        |
| C9  | Impact of Intervals | Show the average impact of intervals as beeswarm.                                            | open              | M6        |
| C10  | Impact of 2-D Intervals | Show the average impact of 2-D intervals as heatmap.                                            | open              | M6        |
| C11  | Interval Settings | Allow the user to change settings for intervals (e.g. number of intervals).                                            | open              | M6        |
| C12  | Improved Division of Tasks | Allow the user to focus on a single task (e.g. model training, error exploration, investigation of importance).                                            | open              | M6        |


## Activities

> Major implementation activities for each component.

| ID  | Component                | Description                                                                  | Status | Point Person    |
| --- | ------------------------ | ---------------------------------------------------------------------------- | ------ | --------------- |
| C1  | Confusion Matrix         |                                                                              |        | Noel Kronenberg |
| A1  |                          | Setting up Streamlit application.                                            | done   | Noel Kronenberg |
| A2  |                          | Integrating trained model with application.                                  | done   | Noel Kronenberg |
| A3  |                          | Integrating trained model with Plotly figure.                                | done   | Noel Kronenberg |
| A4  |                          | Integrating Plotly figure with application.                                  | done   | Noel Kronenberg |
| C2  | Evaluation Metrics       |                                                                              |        | Noel Kronenberg |
| A1  |                          | Calculation of evaluation metrics for trained model.                         | done   | Noel Kronenberg |
| A2  |                          | Displaying of evaluation metrics on Streamlit application.                   | done   | Noel Kronenberg |
| A3  |                          | Adding a bar chart for the comparison of predicted and actual class counts.  | done   | Aodi Chen       |
| A4  |                          | Adding confusion matrix metrics.                                             | done   | Aodi Chen       |
| A5  |                          | Adding collapsible sections to hide metrics.                                 | done   | Noel Kronenberg |
| A6  |                          | Preselecting class with lowest accuracy.                                     | done   | Noel Kronenberg |
| C3  | Data Upload              |                                                                              |        | Noel Kronenberg |
| A1  |                          | Adding form for uploading data.                                              | done   | Noel Kronenberg |
| A2  |                          | Increasing maximum Streamlit upload limit.                                   | done   | Noel Kronenberg |
| C4  | Hosting                  |                                                                              |        | Noel Kronenberg |
| A1  |                          | Setting up Streamlit Community Cloud.                                        | done   | Noel Kronenberg |
| A2  |                          | Making application compatible with hosting.                                  | done   | Noel Kronenberg |
| C5  | Dynamic Model Training   |                                                                              |        | Noel Kronenberg |
| A1  |                          | Adding form for adjustment of parameters.                                    | done   | Noel Kronenberg |
| A2  |                          | Adding function to dynamically train model with new parameters.              | done   | Noel Kronenberg |
| C6  | Faster UX                |                                                                              |        | Noel Kronenberg |
| A1  |                          | Adding caching functionality of data (e.g. uploaded data).                   | done   | Noel Kronenberg |
| A2  |                          | Adding caching functionality of resources (e.g. trained model).              | done   | Noel Kronenberg |
| A3  |                          | Adding data loader signs for transparent loading processes.                  | done   | Noel Kronenberg |
| C7  | Class Selection          |                                                                              |        | Noel Kronenberg |
| A1  |                          | Adding form for selection of class.                                          | done   | Noel Kronenberg |
| A2  |                          | Highlighting class in confusion matrix.                                      | done   | Noel Kronenberg |
| A3  |                          | Adding evaluation metrics for selected class.                                | done   | Noel Kronenberg |
| A4  |                          | Adding highlight to evaluation metrics for selected class.                   | done   | Noel Kronenberg |
| C8  | Overview of Importance Scores | | | Aodi Chen |
| C9  | Impact of Intervals | | | Fabian Henning |
| C10  | Impact of 2-D Intervals | | | Aodi Chen |
| C11  | Interval Settings | | | |
| C12  | Improved Division of Tasks | | | Noel Kronenberg|
