import string
import re
from nltk.corpus import stopwords
from collections import Counter
from os import listdir
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter
from pubmed_lookup import Publication 
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from nltk.corpus import stopwords
from keras.models import load_model
import tensorflow as tf
from keras.models import model_from_json
import os


read_file = pd.read_excel ("MOESM1.xlsx")

read_file.to_csv ("Test.csv",  
                  index = None, 
                  header=True)

df = pd.DataFrame(pd.read_csv("Test.csv"))

x = df.iloc[:, : -1]

y = df.iloc[:, -1]

counter = Counter(y)
print(counter)

plt.figure()
plt.scatter(x['PMID'], y, color = 'red')
plt.title('PMID vs Status')
plt.xlabel('PMID')
plt.ylabel('Status')
plt.show()


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r',encoding='utf8', errors='ignore')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out step words 
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_train):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('sample4')  :
            continue
        if not is_train and not filename.startswith('sample4') and not filename.startswith('sample6') and not filename.startswith('sample7') and not filename.startswith('sample8'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    #load documents
    Included = process_docs('text_project/Included', vocab, is_train)
    Excluded = process_docs('text_project/Excluded', vocab, is_train)
    docs = Included + Excluded
    # prepare labels
    labels = array([0 for _ in range(len(Included))] + [1 for _ in range(len(Excluded))])
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# classify a review as included or excluded
def predict_sentiment(review, vocab, tokenizer,max_length,model):
    # clean review
    line = clean_doc(review, vocab)
    # encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retrive predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return(1-percent_pos), 'EXCLUDED'
    return percent_pos, 'INCLUDED'

# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png',show_shapes=True)
    return model


# load the vocabulary
vocab_filename = 'vocab_project.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
#load training data 
train_docs, ytrain = load_clean_dataset(vocab, True)
# creat the tokenizer
tokenizer = create_tokenizer(train_docs)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum lenth: %d' % max_length)
# encode data 
Xtrain = encode_docs(tokenizer, max_length, train_docs)
#define model
model = define_model(vocab_size, max_length)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# save the model
#model.save(model.h5)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")





























 