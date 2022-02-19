# Author Classifier
## Introduction
This project trains a bert model on detecting the author from a piece of text. And uses the model in make predictions on new data usinf `batch_inference.py`.

## Results.csv
You can find the results.csv in the `data` folder with the required format

## Manging Configurations
All configurations are in the conf folder, you can change the data path, model path, etc. And you can also provide the configuration flag while running the script. You can write --help after the python command to see which configs you can change. Example: `python3 batch_inference.py --help`. Also please notice the path you write will be relevant to the `Abdullah-Tarek/outputs` folder. You can write the full path if that will help you.

## Installation
### Set up
- `$ git https://git.toptal.com/screening/Abdullah-Tarek.git`
- `$ cd Abdullah-Tarek`
- Move the train.csv and test.csv in the `data` folder
- Download the trained model from [here](), unzip it and move it in the `saved_models` fodler.

### Python
- `$ pip install -r requirements.txt`
- `$ Copy the training or testing dataset in the "data" folder `
- `$ python training.py` or `$ python batch_inference.py`

### Docker
- `$ docker build . -t author_classifier`
- `$ docker run -it -v $DATA_FOLDER:/app/data -v $LOCAL_SAVED_MODEL_FOLDER:/app/saved_models author_classifier python batch_inference.py` or `python training.py`