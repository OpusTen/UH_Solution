import pandas as pd
import numpy as np
from tqdm import tqdm
import regex as re
import keras
import h5py
import cPickle
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

#----------------Training-----------------#

#data load
raw = pd.read_csv("Corpus.csv")
raw.info()

#dropping nulls and reset index
raw = raw.dropna()
raw = raw.reset_index(drop=True)

#converting text column_to_list
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


'''clean_data function removes the special characters
    and returns a clean string'''
def clean_data(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower()
    #print re.sub(strip_special_chars, "", string.lower())
    return re.sub(strip_special_chars, "", string.lower())

#appending cleaned up string to a list
clean_list = []
for i in range(0, len(x_list)-1, 1):
    clean_list.append(clean_data(x_list[i]))


'''sent_to_tensor function reads a string in the corpus, and for every word in 
    the string, it fetches the word vector from the twitter data source'''
def sent_to_tensor(string):
    sent_tensor = []
    for word in string.split(' '):
        try:
            sent_tensor.append(vectors[words.index(word)])
        except Exception as e:
            print str(e)
    return np.array(sent_tensor)


#collecting vectors from twitter data for the corresponding words in corpus
tensor_list = []
for sent in clean_list:
    tensor_list.append(sent_to_tensor(sent))


#saving word vectors of the corpus - for re use in future
cPickle.dump(tensor_list, open('tensor_list.p', 'wb'))


#loading up the word vectors
obj = cPickle.load(open('tensor_list.p', 'rb'))

#Histogram function to obtain timestep size ( sentence with max no.of words)
temp_list = [0] * 500
for i in range(0,len(obj)-1,1):
    #print obj[i].shape[0]
    temp_list[obj[i].shape[0]] += 1
temp_list_new = np.array(temp_list)
#print np.argmax(temp_list_new)



#parameters
timesteps = 64
input_dim = 50

#converting all vectors to a finite size
result = pad_sequences(obj, maxlen=timesteps, dtype='float32', padding='post')


#each sentence will be a 1024 dimension vector
latent_dim = 256


#model configuration
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
decoder = Model(inputs, decoded)

sequence_autoencoder.compile(optimizer='rmsprop',loss='mean_absolute_error')

sequence_autoencoder.fit(result,result, batch_size=32,nb_epoch=100)

sequence_autoencoder.save('my_model.h5')



#------------------Predictions----------------#
encoderPredictions = encoder.predict(result)

query = 'jobs in data science'

'''query_transformation function transforms the user query
    to a vector which can be fed to the encoder'''
def query_transformation(string):
    cleaned_query = clean_data(string)
    query_tensor = [sent_to_tensor(cleaned_query)]
    query_vector = pad_sequences(query_tensor, maxlen=timesteps, dtype='float32', padding='post')
    return query_vector

query_vector = query_transformation(query)


'''similarity function will calculate the cosine similarity between
    user query and corpus, prints the top 10 similar documents
    in the corpus'''
def similarity(query_vector, encoderPredictions, topN):
    encoded_query = encoder.predict(query_vector)
    encoded_query_matrix = np.array([encoded_query] * len(obj)).squeeze(axis=1)
    output = cosine_similarity(encoded_query_matrix, encoderPredictions)
    single_out = -1*(output[0, :])
    top_10 = single_out.argsort()[:topN]
    for item in top_10:
        print x_list[item],single_out[item]*-1
        print encoded_query[0, :10]
        print encoderPredictions[item, :10]
        print '\n'
        #break
    return

similarity(query_vector, encoderPredictions, 10)