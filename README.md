# NLP Classifier
## Introduction
This project trains a bert model on any NLP classifcation model. And uses the model in make predictions on new data using `batch_inference.py`. This architecture can be easily extended to cover a lot more models.

## Installation
### Set up
- `$ https://github.com/abdullahtarek/nlp_classifier.git`
- `$ cd nlp_classifier.git`
- Move the train.csv and test.csv in the `data` folder

### Python
- `$ pip install -r requirements.txt`
- `$ Copy the training or testing dataset in the "data" folder `
- `$ python training.py` or `$ python batch_inference.py`

### Docker
- `$ docker build . -t nlp_classifier`
- `$ docker run -it -v $DATA_FOLDER:/app/data -v $LOCAL_SAVED_MODEL_FOLDER:/app/saved_models nlp_classifier python batch_inference.py` or `python training.py`

## Extra options
### Manging Configurations
* All configurations are in the conf folder where you can change the data path, model path, etc. 
* You can also provide the configuration flag while running the script. You can write --help after the python command to see which configs you can change. Example: `python3 batch_inference.py --help`.
