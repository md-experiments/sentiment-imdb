import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import os
import tarfile
import pickle
from utils import *

predictor=predictors()

class SentimentAnalyser():
    def __init__(self,vectorizer='TfIdf',classfier='logistic', 
                data_size=-1, persist=False, cold_start=False, verbose=0):
        self.model_params={'vectorizer': vectorizer,'classifier':classfier}
        self.data_size=data_size
        self.persist_results=persist
        self.cold_start=cold_start
        self.verbose = verbose


    def load_pipe_components(self):
        # These are components of the pipeline such as the choice of Bag of Words vs Tf-Idf and the choise of classifier
        if self.model_params['vectorizer']=='bow':
            vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
        elif self.model_params['vectorizer'].lower()=='tfidf':
            vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)
        else:
            print('Expect vectorizer either "bow" or "TfIdf", got', self.model_params['vectorizer'])

        if self.model_params['classifier']=='logistic':
            classifier = LogisticRegression(solver='lbfgs')
        elif self.model_params['classifier']=='GBM':
            classifier = GradientBoostingClassifier()
        #elif self.model_params['classifier']=='LSTM':
        #    #placeholder for LSTM classifier
        #    classifier = LogisticRegression()
        else:
            print('Expect classifier "logistic", "GBM" , got', self.model_params['classifier'])       
        return vectorizer, classifier

    def train(self,):
        # Trains model by splitting training data into test and train, if persist = True will save locally to a pickled pipeline
        df_tr, _=data_loader(data_size=self.data_size)
  
        X_train, X_test, y_train, y_test = train_test_split(df_tr.Comment, 
                                                            df_tr.Label, test_size=0.33, random_state=42)

        print('Training model vectorizer {0} classifier {1} ' \
                .format(self.model_params['vectorizer'],self.model_params['classifier'])) if self.verbose>0 else ''
        
        vectrz, classfir=self.load_pipe_components()
        
        # Create pipeline using Bag of Words
        self.pipe = Pipeline([("cleaner", predictor),
                        ('vectorizer', vectrz),
                        ('classifier', classfir)])
        # model generation
        self.pipe.fit(X_train,y_train)


        if self.persist_results:
            pickle_name='model_vctrz_'+self.model_params['vectorizer']+'_classf_'+self.model_params['classifier']+'.p'
            pickle.dump(self.pipe, open(pickle_name,'wb'))
        print('Evaluate train set') if self.verbose>0 else ''
        self.eval(X_train,y_train)
        print('Evaluate dev set') if self.verbose>0 else ''
        self.eval(X_test,y_test)

    def eval(self, x,y):
        # Evaluates teh model based on model_params and assuming model is already trained. 
        pickle_name='model_vctrz_'+self.model_params['vectorizer']+'_classf_'+self.model_params['classifier']+'.p'
        if self.cold_start or not os.path.exists(pickle_name):
            print('Training Model')
            self.train()
        with open(pickle_name,'rb') as s1:
            pipe=pickle.load(s1)
        
        predicted = pipe.predict(x)
        print("Model Accuracy:",metrics.accuracy_score(y, predicted))  if self.verbose>0 else ''

    def gen_lbl(self, out, format='ls'):
        if not list(set(out)  - {0, 1})==[]:
            print('Too many labels, labels:', str(set(out) ))
        output = ['aclImbd/train/pos' if ent ==1 else 'aclImbd/train/neg' for ent in out]

        return output

    def predict(self, query, ):
        if type(query)==str:
            query=[query]
            ret_out='str'
        elif type(query)==list:
            ret_out='ls'
        else:
            print('Inputs need to be list of strings or single string')
            sys.exit()
        pickle_name='model_vctrz_'+self.model_params['vectorizer']+'_classf_'+self.model_params['classifier']+'.p'
        if self.cold_start or not os.path.exists(pickle_name):
            print('Training Model')
            self.train()

        with open(pickle_name,'rb') as s1:
            self.pipe=pickle.load(s1)
        
        out=self.pipe.predict(query)
        output=self.gen_lbl(out, format=ret_out) 
        return output

if __name__ == "__main__":
    import sys
    file_path=sys.argv[1]
    with open(file_path,'r') as f:
        query=f.readlines()
    print('Evaluating', [qry for qry in query])
    sa=SentimentAnalyser()
    print(sa.predict(query))
    quit()
