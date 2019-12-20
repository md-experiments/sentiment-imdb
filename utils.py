import datetime
import os, sys
import pandas as pd
import tarfile
import spacy
import string

from sklearn.base import TransformerMixin
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

#nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()
# Custom transformer using spaCy

class predictors(TransformerMixin):
    # Transforms all to lower case
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [text.strip().lower() for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}



# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

def load_from_tar(file_name, path_to_load, extracted=True):
    # Loads data from files either from 'exctacted' .tar or from the archived file itself
    # file_name only used in extracted = False
    dt0=datetime.datetime.now()
    comments=[]
    ratings=[]
    if extracted:
        files_tr_pos=os.listdir(path_to_load)
    else:
        tar = tarfile.open("aclImdb_v1.tar")
        tar_memb=tar.getmembers()
        files_tr_pos=[memb.name for memb in tar_memb if path_to_load in memb.name]
    
    for memb in files_tr_pos:
        if extracted:
            f=open(path_to_load+memb,'r')
            content=f.read()
            comment=content
        else:
            f=tar.extractfile(memb)
            content=f.read()
            comment=content.decode("utf-8") 

        comments.append(comment)
        rating=int(memb.split('_')[1].split('.')[0])
        ratings.append(rating)
        if extracted:
            f.close()     
    if not extracted:
        tar.close()
    df=pd.DataFrame([])
    df['Comment']=comments
    df['Ratings']=ratings
    print('Time taken', path_to_load, datetime.datetime.now()-dt0)
    return df


def data_loader(from_local=True, data_size=-1):
    # Tries to run data from locally saved collections from the raw file. If those do not exist, then will attempt from
    # extracted files finally if neither available will try directly from raw .tar file
    if from_local & \
        os.path.exists('./local_data/train_data.csv') & \
        os.path.exists('./local_data/test_data.csv'):
        # Read from .csv if already locally loaded
        df_tr=pd.read_csv('./local_data/train_data.csv', index_col=0)
        df_ts=pd.read_csv('./local_data/test_data.csv', index_col=0)
    else:    
        to_be_saved=True
        if not os.path.exists('./local_data/'):
            os.mkdir('./local_data/')        
        if os.path.exists('./aclImdb/train/pos/') & \
            os.path.exists('./aclImdb/train/neg/') & \
            os.path.exists('./aclImdb/test/neg/') & \
            os.path.exists('./aclImdb/test/neg/'):
            extracted=True

        else:
            extracted=False

            if not os.path.exists("aclImdb_v1.tar"):
                sys.exit('Please download "aclImdb_v1.tar" and place it in this directory')

        print('Loading files from raw')
        df_tr_pos=load_from_tar("aclImdb_v1.tar", 'aclImdb/train/pos/',extracted=extracted)
        df_tr_neg=load_from_tar("aclImdb_v1.tar", 'aclImdb/train/neg/',extracted=extracted)
        df_ts_pos=load_from_tar("aclImdb_v1.tar", 'aclImdb/test/pos/',extracted=extracted)
        df_ts_neg=load_from_tar("aclImdb_v1.tar", 'aclImdb/test/neg/',extracted=extracted)
        df_tr=df_tr_pos.append(df_tr_neg,ignore_index=True)
        df_ts=df_ts_pos.append(df_ts_neg,ignore_index=True)
        if to_be_saved:
            df_tr.to_csv('./local_data/train_data.csv')
            df_ts.to_csv('./local_data/test_data.csv')
    if data_size>0:
        df_tr=df_tr.sample(data_size, random_state=0,)
        df_ts=df_ts.sample(data_size, random_state=0,)
    df_tr['Label']=df_tr.Ratings.apply(lambda x: 1 if int(x)>5 else 0)
    df_ts['Label']=df_ts.Ratings.apply(lambda x: 1 if int(x)>5 else 0)
    return df_tr, df_ts