import numpy as np
import random
import torch

# tomer's code for a seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_file(path):
    '''
    :param path: full path for a tagged file
    :return: data - a list of all the sentences(lists), each word as
    a 2 position list - word and tag
    '''
    data = []
    with open(path, 'r') as file:
        current_sentence = []
        for line in file:
            if line != '\n':
                current_word = line.replace('\n', '').split("\t")
                if current_word[1][0] == 'O':
                    current_word[1] = 0
                else:
                    current_word[1] = 1
                current_sentence.append(current_word)
            else:
                data.append(current_sentence)
                current_sentence = []
    return data

def data_to_vectors(data, glove):
    '''
    :param data: data of words created by read_file
    :return: the data with vectorized glove representation for the words
    '''
    vec_data = data.copy()
    for sen in vec_data:
        for word in sen:
            if word[0].lower() not in glove.key_to_index:
                word[0] = (np.random.random(200) * 2 - 1)
            else:
                word[0] = glove[word[0].lower()]
    return vec_data

def add_context(vec_data, dot_repr):
    '''
    :param vec_data: data as created by data_to_vectors
    :param dot_repr: the glove representation of '.'
    :return: vec_data with each data vector surrounded by its' neighbors
    '''
    vec_data_with_context = vec_data.copy()
    for sen in vec_data:
        for i, word in enumerate(sen):
            if i != 0 and i != (len(sen) - 1):
                word[0] = np.concatenate(
                    (sen[i - 1][0], sen[i - 1][0], sen[i + 1][0]))
            if i == 0 and i != (len(sen) - 1):
                word[0] = np.concatenate(
                    (dot_repr, sen[i - 1][0], sen[i + 1][0]))
            if i != 0 and i == (len(sen) - 1):
                word[0] = np.concatenate(
                    (sen[i - 1][0], sen[i - 1][0], dot_repr))
            if i == 0 and i == (len(sen) - 1):
                word[0] = np.concatenate(
                    (dot_repr, sen[i - 1][0], dot_repr))
    return vec_data_with_context

def sep_X_y(vec_data):
    '''
    :param vec_data: data as created by data_to_vectors
    :return: X: feature matrix
             y: labels vector
    '''
    X = []
    y = []
    for sen in vec_data:
        for word in sen:
            X.append(word[0])
            y.append(word[1])
    return X, y


