import gensim
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

'''This python file contain functions used to read the word embeddings of choice to assist the training process'''

def get_UD_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    '''
    This function reads the urban dictionary embeddings - explained in the documentation
    Args:
        filename: the path to the urban dictionary word embeddings - string
        word_dictionary: python dictionary of the individual words in the training dataset
        vocab_size: the number of unique words in the training dataset - integer
        embedding_dimension: the embeddings size , as explained in the documetnation - integer

    Returns:
            embedding_matrix: a numpy matrix of each word in the training dataset and
            its matching vector in the urban dictionary word embedding

    '''
    UD_model = gensim.models.KeyedVectors.load_word2vec_format(filename)
    embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in UD_model.wv.vocab:
            embedding_vector = UD_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_Glove_embeddings(filename, word_dictionary, embedding_dimension):
    '''
    This function reads the glove embeddings - explained in the documentation
    Args:
        filename: the paht to the glove word embeddings - string
        word_dictionary: python dictioanry of the individual words in the training dataset
        vocab_size: the number of unique words in the training dataset - integer
        embedding_dimension: the embeddings size , as explained in the documetnation - integer

    Returns:
            embedding_matrix: a numpy matrix of each word in the training dataset and
            its matching vector in teh glove word emebddings

    '''
    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    glove_embedding_matrix = np.zeros((len(word_dictionary) + 1, embedding_dimension))
    for word, i in word_dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            glove_embedding_matrix[i] = embedding_vector

    return glove_embedding_matrix

def get_sswe_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    '''
    This function reads the SSWE embeddings - explained in the documentation
    Args:
        filename: the paht to the urban dictionary word embeddings - string
        word_dictionary: python dictioanry of the individual words in the training dataset
        vocab_size: the number of unique words in the training dataset - integer
        embedding_dimension: the embeddings size , as explained in the documetnation - integer

    Returns:
            embedding_matrix: a numpy matrix of each word in the training dataset and
            its matching vector in teh SSWE word emebddings

    '''
    sswe_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    sswe_embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in sswe_model.wv.vocab:
            embedding_vector = sswe_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            sswe_embedding_matrix[i] = embedding_vector
    return sswe_embedding_matrix

def get_google_news_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    '''
    This function reads the word2vec embeddings - explained in the documentation
    Args:
        filename: the paht to the urban dictionary word embeddings - string
        word_dictionary: python dictioanry of the individual words in the training dataset
        vocab_size: the number of unique words in the training dataset - integer
        embedding_dimension: the embeddings size , as explained in the documetnation - integer

    Returns:
            embedding_matrix: a numpy matrix of each word in the training dataset and
            its matching vector in the word2vec word emebddings

    '''
    sswe_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    sswe_embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in sswe_model.wv.vocab:
            embedding_vector = sswe_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            sswe_embedding_matrix[i] = embedding_vector
    return sswe_embedding_matrix
