
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
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    #conert to lowercase
    tokens = [word.lower() for word in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out step words 
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

#load the document
"""
filename = 'text_project/Included/sample8.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)
"""
#load the doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    #load documnet
    doc = load_doc(filename)
    #clean documnet
    tokens = clean_doc(doc)
    #update counts
    vocab.update(tokens)
    
#load all documnets in a directory
def process_docs(directory, vocab):
    #walk through all files in the folder
    for filename in listdir(directory):
        if filename.startswith('sample4') and filename.startswith('sample5') and filename.startswith('sample6') and filename.startswith('sample7') and filename.startswith('sample8') :
            continue
       
        #create the full path of the file to open
        path = directory+'/'+ filename
        #add document to vocab
        add_doc_to_vocab(path, vocab)

#define vocab 
vocab = Counter()
#add all documents to vocab
process_docs('text_project/Included', vocab)
process_docs('text_project/Excluded', vocab)
#print the size of the vocab
print(len(vocab))
#print the top wordds in the vocab
print(vocab.most_common(50))
 
#save list to file
def save_list(lines, filename):
    #convert lines to a single blob of text
    data = '\n'.join(lines)
    #open file
    file = open(filename, 'w',encoding='utf8', errors='ignore')
    # write text
    file.write(data)
    #close file
    file.close()
    
#keep tokens with a minimum occurrence
min_occurrence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))
#save tokens to a vocabulary file 
save_list(tokens, 'vocab_project.txt')

 
  
    
    
    
    
    
    
    
    
    
    
    
    














