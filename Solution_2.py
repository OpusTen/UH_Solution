import pandas as pd
import numpy as np
from tqdm import tqdm
import regex as re
from sklearn.metrics.pairwise import cosine_similarity

#----------------Training-----------------#

#data load
raw = pd.read_csv("Corpus.csv")
raw.info()

#dropping nulls and reset index
raw = raw.dropna()
raw = raw.reset_index(drop=True)

x_list = raw['text'].tolist()


'''loadGlove function reads a text file which contains
    twitter data to return a word and the corresponding 50 Dimension Vector.
    Data is downloaded from the following link :
    https://nlp.stanford.edu/projects/glove/'''

def loadGlove():
    words, vectors = [], []
    with open('/Users/hancel/Documents/Weekend Project/glove.twitter.27B/glove.twitter.27B.50d.txt', 'rb') as f:
        for line in tqdm(f.readlines()):
             line = line.split()
             word = line[0]
             vector = [float(x) for x in line[1:]]
             words.append(word)
             vectors.append(vector)
    return words, vectors

words, vectors = loadGlove()

#cleaning string
def clean_data(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower()
    return re.sub(strip_special_chars, "", string.lower())


clean_list = []
for i in range(0, len(x_list)-1, 1):
    clean_list.append(clean_data(x_list[i]))


#summing up all the word vectors
def sentenceToTensor(string):
  #string = ['hi', 'my', 'name'...]
  sum_vector = np.zeros((1, 50))
  for word in string.split(' '):
    try:
        vector = vectors[words.index(word)]
        sum_vector += vector
        #print sum_vector.shape
    except:
        pass
  return sum_vector

#generating the list
tensor_list = []
for i in tqdm(range(0, len(clean_list)-1, 1)):
    tensor_list.append(sentenceToTensor(clean_list[i]))


#----------------Predictions--------------------#

encoderPredictions = np.array(tensor_list).reshape(-1, 50)

query = 'We are looking for analytics service provider in Bangalore?'

'''query_transformation function transforms the user query
    to a vector which can be fed to the similarity function'''
def query_transformation(string):
    cleaned_query = clean_data(string)
    query_tensor = sentenceToTensor(cleaned_query)
    return query_tensor

query_vector = query_transformation(query)


'''similarity function will calculate the cosine similarity between
    user query and corpus, prints the top 10 similar documents
    in the corpus'''
def similarity(query_vector, encoderPredictions, topN):
    encoded_query_matrix = np.array([query_vector] * 6811).squeeze(axis=1)
    output = cosine_similarity(encoded_query_matrix, encoderPredictions)
    single_out = -1*(output[0, :])
    top_10 = single_out.argsort()[:topN]
    for item in top_10:
        print x_list[item],single_out[item]*-1
        print '\n'
    return

similarity(query_vector, encoderPredictions, 10)