import tensorflow as tf
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional #,Merge
from keras.models import Model,Sequential

from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import L1L2
from transformers import *

'''This python file contains functions that design the different models LR, MLP, LSTM, and BiLSTM'''
def LR(inp_dim):
    '''
    This function design the LR model
    Args:
        inp_dim: the number of the unique features/words in the dataset to be considered for training LR model

    Returns:
            model: a LR model ready for training

    '''
    print("Model LR")
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=inp_dim
                    , kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                    , bias_regularizer=L1L2(l2=0.00000001)))
    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #print(model.summary())
    return model

def MLP(inp_dim, vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
    '''
    This function design MLP model to be ready for training
    Args:
        inp_dim: the length of the sentences in teh training dataset - integer
        vocab_size: the number of unique words in the trianing dataset - ineger
        embed_size: the embeddings dimentiosn size depends o nthe used word emebddings - refer to the documentation - integer
        use_word_embeddings: flag to indicate if word emebddings going to be used or not - boolean
        embedding_matrix: numpy matrix with the word embeddings
        embedding_trainable: flag to indicate if the word emebddings to train with the model or not - boolean

    Returns:
            model: the MLP model ready for the training process
    '''
    model = Sequential()
    if use_word_embeddings == True:
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
        model.add(Flatten())
        model.add(Dropout(0.50))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu', kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001)))
        model.add(Dense(64, activation='relu' , kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                        , bias_regularizer=L1L2(l2=0.00000001)))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                        , bias_regularizer=L1L2(l2=0.00000001)))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    #print(model.summary())
    return model

def lstm_keras(inp_dim, vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
    '''
    This function design LSTM model to be ready for training
    Args:
        inp_dim: the length of the sentences in teh training dataset - integer
        vocab_size: the number of unique words in the trianing dataset - ineger
        embed_size: the embeddings dimentiosn size depends o nthe used word emebddings - refer to the documentation - integer
        use_word_embeddings: flag to indicate if word emebddings going to be used or not - boolean
        embedding_matrix: numpy matrix with the word embeddings
        embedding_trainable: flag to indicate if the word emebddings to train with the model or not - boolean

    Returns:
            model: the LSTM model ready for the training process
    '''
    model = Sequential()
    if use_word_embeddings == True:
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001)))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    #print (model.summary())
    return model

def lstm_bert (embed_size):
    model = Sequential()
    model.add(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001)))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def blstm(inp_dim,vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
    '''
    This function design BiLSTM model to be ready for training
    Args:
        inp_dim: the length of the sentences in teh training dataset - integer
        vocab_size: the number of unique words in the trianing dataset - ineger
        embed_size: the embeddings dimentiosn size depends o nthe used word emebddings - refer to the documentation - integer
        use_word_embeddings: flag to indicate if word emebddings going to be used or not - boolean
        embedding_matrix: numpy matrix with the word embeddings
        embedding_trainable: flag to indicate if the word emebddings to train with the model or not - boolean

    Returns:
            model: the BiLSTM model ready for the training process


    '''
    model = Sequential()
    if use_word_embeddings == True:
        print("use word embedding is True")
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001))))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

