#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import jieba
import jieba.posseg as pseg

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from keras.utils import plot_model
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding, CuDNNGRU
from keras.models import Sequential
from keras.initializers import Constant

from sklearn.model_selection import train_test_split


import gensim


# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[3]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[4]:


def dropcomma(sent):
    if isinstance(sent, str):
        return sent.replace(',', '')
    else:
        return ''


# In[5]:


#  whether or not use key_words in dataset(special for this toutiao dateset)
def add_keywords(x):
    if True:
        return np.array([dropcomma(xi) for xi in x])
    else:
        return np.full_like(x, '')


# ### Data Preprocessing

# In[6]:


# df_aug = pd.read_csv('./data/train_data_preprocessed_with_NA_aug_40000.csv')
df_test = pd.read_csv('./data/test_data.csv') 
df_train_all = pd.read_csv('./data/df_train_all.csv')
#print(len(df_train_all))


# In[7]:


sentences = df_train_all['title'].values
key_words = df_train_all['keyword'].values
labels = df_train_all['label'].values

sentences_t = df_test['title'].values
key_words_t = df_test['keyword'].values


# In[8]:


print(len(sentences))
print(len(labels))


# In[9]:


key_words_drop_comma = add_keywords(key_words)
key_words_drop_comma_t = add_keywords(key_words_t)


# In[10]:


sentences_train = [
    sentence + keyword for sentence, keyword in zip(sentences,key_words_drop_comma) 
    if sentence is not None]


# In[11]:


sentences_test = [
    sentence + keyword for sentence, keyword in zip(sentences_t,key_words_drop_comma_t) 
    if sentence is not None]


# In[12]:


sentences_all = sentences_train + sentences_test


# In[13]:


len(sentences_all)


# In[14]:


def get_sent_list(sentences):
    sent_list = []
    flag_list = ['n','ng','nr','nrfg','nrt','ns','nt','nz','s','j','l','i','v','vn','eng']
    
    for num in range(len(sentences)):
        
        if num % 10000 == 0:
            print("No. %d" % (num))
        
        sent = sentences[num]
        sent = pseg.cut(sent) # get part of speech

        tmp_list = []
        for word, flag in sent:
            if flag in flag_list:
                tmp_list.append(word)

        sent_list.append(tmp_list)
    
    
    return sent_list


# In[15]:


sent_list_train = get_sent_list(sentences_train)
sent_list_test = get_sent_list(sentences_test)
sent_list_all = get_sent_list(sentences_all)


# In[16]:


print(len(sent_list_train))
print(len(sent_list_test))
print(len(sent_list_all))


# In[17]:


with open('./data/sents_test.txt', 'w') as f:
    for item in sent_list_test:
        f.write("%s\n" % item)


# In[18]:


train_and_test_texts = open('./data/sents_all.txt').read().split('\n')
train_texts = open('./data/sents_train.txt').read().split('\n')
test_texts = open('./data/sents_test.txt').read().split('\n')


# In[19]:


len(max(train_and_test_texts, key=len))


# In[20]:


MAX_SEQUENCE_LENGTH = 300

tokenizer_train = Tokenizer()
tokenizer_test = Tokenizer()
tokenizer_all = Tokenizer()

tokenizer_train.fit_on_texts(train_texts[:-1])
tokenizer_test.fit_on_texts(test_texts[:-1])
tokenizer_all.fit_on_texts(train_and_test_texts[:-1])

sequences_train = tokenizer_train.texts_to_sequences(train_texts[:-1]) # only training sentences
sequences_test = tokenizer_train.texts_to_sequences(test_texts[:-1]) # only testing sentences
sequences_all = tokenizer_all.texts_to_sequences(train_and_test_texts[:-1]) # training + testing

word_index = tokenizer_all.word_index

print('Found %s unique tokens.' % len(word_index))

training_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
testing_data = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(labels)

print('Shape of training data tensor:', training_data.shape)
print('Shape of label tensor:', labels.shape)
print('Shape of testing data tensor:', testing_data.shape)


# In[21]:


# Use train_test_split to split our data into train and validation sets

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    training_data, labels, random_state=42, test_size=0.2)


# ### LSTM Using Own Trained Embedding

# In[22]:


# original model
EMBEDDING_DIM = 232

model_1 = Sequential()
model_1.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
          input_length=MAX_SEQUENCE_LENGTH))
model_1.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
model_1.add(Dropout(0.2))
model_1.add(Dense(labels.shape[1], activation='softmax'))
model_1.summary()


# In[23]:


# path = './models/lstm_add_back_epoch_10_model_4.h5'
# model_1 = load_model(path)


# In[24]:


model_1.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['acc'])

print(model_1.metrics_names)

model_1.fit(training_data, labels, 
            epochs=10, batch_size=512) # full data 


# In[26]:


# path = './models/lstm_add_back_epoch_11_model_4.h5'
# model_1.save(path)


# In[25]:


pred_list = []
pred_list_ans = []

pred_list = model_1.predict(testing_data, verbose=1, batch_size=1024)
pred_list_ans = [pred_list[num].argmax() for num in range(len(pred_list))]


# In[27]:


id_list = df_test['id'].values
df_ans = pd.DataFrame(data=pred_list_ans, index=id_list, columns=['label'])
# df_ans.to_csv('./results/lstm_add_back_epoch_11_model_4.csv')


# ### LSTM Using Pretrained Embedding Layer

# In[33]:


print('Indexing word vectors.')

embeddings_index = {}
with open('./data/merge_sgns_bigram_char300.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# In[34]:


# prepare embedding matrix
MAX_NUM_WORDS = 1348401
EMBEDDING_DIM = 300

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[35]:


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In[36]:


# LSTM with no word2vec

model_wv = Sequential()
model_wv.add(embedding_layer)
model_wv.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model_wv.add(Dropout(0.2))
model_wv.add(Dense(labels.shape[1], activation='softmax'))
model_wv.summary()


# In[48]:


model_wv = load_model('./models/lstm_wv2_add_back_epochs_20_full_data.h5')


# In[49]:


model_wv.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['acc'])

print(model_wv.metrics_names)

model_wv.fit(training_data, labels, epochs = 4, batch_size = 1024)

model_wv.save('./models/lstm_wv2_add_back_epochs_24_full_data.h5')


# In[50]:


pred_list = []
pred_list_ans = []
pred_list = model_wv.predict(testing_data, verbose = 1, batch_size = 1024)
pred_list_ans = [pred_list[num].argmax() for num in range(len(pred_list))]


# In[51]:


id_list = df_test['id'].values
df_ans = pd.DataFrame(data=pred_list_ans, index=id_list, columns=['label'])
df_ans.to_csv('./results/lstm_wv2_add_back_epochs_24_full_data.csv')


# In[ ]:




