# Sentiment IMDB

### Introduction
The objective is to provide a module that can perform sentiment analysis based on pre-trained models and have persistance.

The module allows both direct prediction based on pre-trained models as well as the capability to train/predict based on a model of choice

The supported models are Logistic Regression and GBM classifiers. With vectorization done either as Bag-of-Words or TfIdf. The data used for training is from IMDB Reviews (https://ai.stanford.edu/~amaas/data/sentiment/) hence usage is recommended to be limited to similar context. 

Performance of the different pretrained models by data set as follows. This repo is automatically set to run on the best performing set based on Validation performance:

Vectorizer  | Classifier | Train (67% aclImbd/train) | Dev (33% aclImbd/train) | Validation (aclImbd/test)
------------- | -------------|---|---|---
BoW  | Logistic | 99.8% | 87% | 85.3%
BoW  | GBM | 81.7% | 79.8% | 79.9%
Tf-Idf  | Logistic | 93.4% | 87.6% | 86.2%
Tf-Idf  | GBM | 82.7% | 80.1% | 80.1%

### Installation
1. Download repo locally
2. Download training data (needed for retrain and running some of the tests) to the repo from https://ai.stanford.edu/~amaas/data/sentiment/ 
Folder structure should be as follows:

	    .
	    ├── tests/
	    ├── aclImdb/ # Data file extract
	    ├── aclImdb_v1.tar # Raw downloaded data file
	    ├── main.py
	    ├── tests.py
	    ├── utils.py
	    ├── ...
    
3. Run `pip install requirements.txt`

### Usage
##### From Terminal
Navigate to the project directory. Can predict sentiment by pointing to a simple text file containing one comment per line:
`python main.py "./tests/test.txt"`
##### From Python
Navigate to the project directory. After importing SentimentAnalyzer one can choose type of model, size of dataset to use and verbosity when setting up
```
from main import SentimentAnalyser
sa=SentimentAnalyser(
	vectorizer='TfIdf', # Specify TfIdf or BoW
	classfier='logistic',  # Specify logistic or GBM
	data_size=-1, # Specify explicit number of entries from data used for training 
	# (irrelevant for .predict). -1 means use all data
	
	persist=False, # If model train is called, new trained model will be persisted
	cold_start=False, # .predict retrains model on the spot
	verbose=0) # Levels of verbosity

sa.train() # Trains and saves locally if persisted
query=['Comment', 'more comment', 'Three comment']
sa.predict(query) # Uses selected model to predict for list of strings
```

### Feature List & Next Steps
- Data Science
- [x] BoW / TfIdf Vectorizer
- [x] Logistic & GBM
- [x] Compare models based on train / dev / validation set
- [ ] Expand to LSTM
- [ ] Hyperparameter optimisation needed for GBM to improve performance
- Module Features
- [x] Set size of data set load to the entire class
- [x] Change and retrain models on demand
- [x] Persist models to .pickle
- [ ] Improve to more flexible method for persistance than pickle
- [x] Store to github and comment purpose of each function
- [ ] When persist = False, Analyzer still uses the last saved model to eval/predict, ideally it should be the current pipe and not loaded pipe, TBD
- Tests
- [ ] Validate each component of the NLP pipe works standalone
- [x] Validate all models can train and evaluate .train() & .eval()
- [x] Validate models can predict .predict()

