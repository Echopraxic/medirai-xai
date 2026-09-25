# Model Development
To recreate the modelling results, create a python virtual environment and install the `requirements.txt`.

Development Details:
- Python version == 3.12
- Development OS == MacOS 14.6.1

# Iteration 0: Vanilla Models

## Overview:
- 3-layer CNN trained on the melonoma dataset only.
- Performed well on validation data, but poorly on synthetic IRL data.
- To retrain the model run `main.py`. This will fetch the data from source and train the model. 
