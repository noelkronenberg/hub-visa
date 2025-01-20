# Local install instructions

The project uses Python 3 and some data analysis packages.

## Install Miniconda

**This step is only necessary if you don't have conda installed already**:

- download the Miniconda installer for your operating system (Windows, MacOSX
  or Linux) [here](https://docs.conda.io/en/latest/miniconda.html)
- run the installer following the instructions
  [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
  depending on your operating system.

## Create conda environment

```sh
# Clone this repo
git clone https://github.com/noelkronenberg/hub-visa
cd hub-visa
# Create a conda environment with the required packages for this tutorial:
conda env create -f environment.yml
```

## Check your install

To make sure you have all the necessary packages installed, we **strongly
recommend** you to execute the `check_env.py` script located at the root of
this repository:

```sh
# Activate your conda environment
conda activate semester_project_group_a
python check_env.py
```

Make sure that there is no `FAIL` in the output when running the `check_env.py`
script.

## Open web app

To build a live preview of the actual web app, execute the following skript from the root of this repository:

```sh
streamlit run app/app.py
```
