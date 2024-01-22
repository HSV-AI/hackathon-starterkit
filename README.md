# hackathon-starterkit
This project provides a starter kit that can be used to bootstrap a hackathon project. We have selected some basic libraries that will cover many of the AI related tasks that come up in hackathon challenges.

- jupyter
- matplotlib
- pandas
- scikit-learn
- streamlit
- torch
- transformers
- wandb

## Installation

This project uses conda to provide an easy way to install python and create a local environment for your hackathon project. This project uses python 3.11, which is the latest at the time supported by PyTorch.

```conda create --name your_environment_name python=3.11```

To activate your environment, run 
```conda activate your_environment_name```

Now you can install the dependencies for this project with

```pip install -r requirements.txt```

## Example notebooks

There are example notebooks in the notebooks directory that include:
- Introduction to Jupyter Notebooks
- Data Analysis with matplotlib
- Displaying geographic data
- NLP topic modeling
- Scikit Learn for Classification

