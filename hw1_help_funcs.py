import numpy as np
import random
import torch
from gensim import downloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sys
import pickle
from sklearn import svm
from sklearn.metrics import f1_score
from gensim.models import Word2Vec


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

def data_to_vectors_extra_features(data1, data2, glove):
    '''
    :param data1: data of words created by read_file
    :param data1: data of words created by read_file
    :return: data1 with vectorized glove representation for the words.
    if not in glove, use the the vector found in the word2vec model trained
    '''
    vec_data = data1.copy()
    train_sentences = [[word[0].lower() for word in sen] for sen in data1]
    dev_sentences = [[word[0].lower() for word in sen] for sen in data2]
    sentences = train_sentences + dev_sentences
    model = Word2Vec(sentences=sentences, vector_size=200, window=5,
                     min_count=1, workers=4, epochs=100)
    for i, sen in enumerate(vec_data):
        for j, word in enumerate(sen):
            to_add = []
            # has capital letters
            to_add.append(float(word[0].lower() != word[0]))
            # all capital letters
            all_capital_flag = True
            for let in word[0]:
                if let.lower() == let:
                    all_capital_flag = False
                    break
            to_add.append(float(all_capital_flag))
            # info about the word before
            if j == 0 or i == 0:
                to_add.append(0)
                to_add.append(0)
            else:
                to = float(sen[j-1][0] == glove['to'])
                at = float(sen[j-1][0] == glove['at'])
                on = float(sen[j-1][0] == glove['on'])
                _in = float(sen[j-1][0] == glove['in'])
                qu = float(sen[j - 1][0] == glove["'"])
                dots = float(sen[j - 1][0] == glove[':'])
                to_add.append(4*at + 3*to + 2*on + 1*_in)
                to_add.append(2*dots + 1*qu)
            if j == len(sen)-1:
                to_add.append(0)
            else:
                won = float(sen[j+1][0].lower() == 'won')
                to_add.append(won)
            if word[0].lower() not in glove.key_to_index:
                if word[0].lower() not in model.wv:
                    print("not suppose to happen...") #TODO remove
                    word[0] = (np.random.random(200) * 2 - 1)
                else:
                    word[0] = model.wv[word[0].lower()]
            else:
                word[0] = glove[word[0].lower()]
            word[0] = np.concatenate((word[0], to_add))
    return vec_data



# def add_context(vec_data, dot_repr):
#     '''
#     :param vec_data: data as created by data_to_vectors
#     :param dot_repr: the glove representation of '.'
#     :return: vec_data with each data vector surrounded by its' neighbors
#     '''
#     vec_data_with_context = vec_data.copy()
#     for sen in vec_data:
#         for i, word in enumerate(sen):
#             if i != 0 and i != (len(sen) - 1):
#                 word[0] = np.concatenate(
#                     (sen[i - 1][0], sen[i - 1][0], sen[i + 1][0]))
#             if i == 0 and i != (len(sen) - 1):
#                 word[0] = np.concatenate(
#                     (dot_repr, sen[i - 1][0], sen[i + 1][0]))
#             if i != 0 and i == (len(sen) - 1):
#                 word[0] = np.concatenate(
#                     (sen[i - 1][0], sen[i - 1][0], dot_repr))
#             if i == 0 and i == (len(sen) - 1):
#                 word[0] = np.concatenate(
#                     (dot_repr, sen[i - 1][0], dot_repr))
#     return vec_data_with_context

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

class Network(nn.Module):
   def __init__(self, input_dim=200):
       super().__init__()
       self.hidden_dim = 10
       self.layer_1 = torch.nn.Linear(input_dim, self.hidden_dim)
       self.layer_2 = torch.nn.Linear(self.hidden_dim, 1)
       self.activation = F.relu

   def forward(self, x):
       x = self.layer_1(x)        # x.size() -> [batch_size, self.hidden_dim]
       x = self.activation(x)     # x.size() -> [batch_size, self.hidden_dim]
       x = self.layer_2(x)        # x.size() -> [batch_size, 1]
       x = torch.sigmoid(x)
       return x

def train_nn(net,train_loader, valid_loader, clip=1000 ,epochs=10, print_every=1000):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()
    counter = 0
    for e in range(epochs):
        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            # x.size() -> [batch_size]
            batch_size = inputs.size(0)
            # IMPORTANT - change the dimensions of x before it enters the NN, batch size must always be first
            x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
            x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
            predictions = net(x)

            # calculate the loss and perform backprop
            loss = loss_func(predictions.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                net.eval()
                print_flag = True
                for inputs, labels in valid_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history

                    # get the output from the model
                    # x.size() -> [batch_size]
                    batch_size = inputs.size(0)
                    # IMPORTANT - change the dimensions of x before it enters the NN, batch size must always be first
                    x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                    x = x.view(batch_size,
                               -1)  # x.size() -> [batch_size, 1]
                    val_predictions = net(x)
                    val_loss = loss_func(val_predictions.squeeze(),
                                         labels.float())
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
    return net


def add_context(vec_data, context_size, sen_separator):
    """
    :param vec_data: data read by data_to_vectors...
    :param context_size: size of context from each size
    :param sen_separator: the vector separating between sentences
    :return: the data with its context
    """
    vec_length = vec_data[0][0][0].shape[0]
    length = context_size
    for sen in vec_data:
        for i, word in enumerate(sen):
            length += 1
            if i == len(sen) - 1:
                length += context_size
    # print(f'length is {length}')
    one_vec_data = np.zeros(length*vec_length)
    jump, index = 0, 0
    non_data_idx = []
    # for i in range(context_size):
    #     np.put(one_vec_data,
    #            np.arange(len(one_vec_data) - (i+1) * vec_length - len(one_vec_data) - i * vec_length),
    #            sen_separator)
    for i, sen in enumerate(vec_data):
        for j, word in enumerate(sen):
            if j == 0:
                for i in range(context_size):
                    non_data_idx.append(index+jump)
                    np.put(one_vec_data, np.arange(vec_length*(index+jump),
                                                   vec_length*(index+jump)+vec_length), sen_separator)
                    jump += 1

            np.put(one_vec_data, np.arange(vec_length*(index+jump), vec_length*(index+jump)+vec_length), word[0])
            index += 1
    # print(f'jump is {jump}')
    non_data_add = [length-k for k in range(context_size, 0, -1)]
    # print(non_data_add)
    non_data_idx += non_data_add
    # print(non_data_idx)
    return [one_vec_data[
            vec_length * i - context_size * vec_length:vec_length * i + (
                        context_size + 1) * vec_length]
            for i in range(1, len(one_vec_data) // vec_length - 1) if
            not (i in non_data_idx)]


def agg_context_mean(X, context_size, vec_length):
    X_new = []
    for x in X:
        x_mean_start, x_mean_end = np.zeros(vec_length), np.zeros(vec_length)
        for i in range(context_size):
            x_mean_start += x[i*vec_length: (i+1)*vec_length]
            x_mean_end += x[len(x)-(i+1)*vec_length: len(x)-i*vec_length]
        x_mean_start, x_mean_end = x_mean_start/context_size, x_mean_end/context_size
        X_new.append(np.concatenate((x_mean_start, x[context_size*vec_length: (context_size + 1)*vec_length], x_mean_end)))
    return X_new


if __name__ == '__main__':
    set_seed()
    ####################### To Remove
    with open('glove.pickle', 'rb') as handle:
        glove = pickle.load(handle)
    #######################

    _, train_filename, dev_filename = sys.argv
    # train_filename, = r'..\data\train.tagged'
    # dev_filename = r'..\data\dev.tagged'
    print(train_filename, dev_filename)


    # GLOVE_PATH = 'glove-twitter-200' # ToDo - retuen to code
    # glove = downloader.load(GLOVE_PATH)

    train_data = data_to_vectors(read_file(train_filename), glove)
    dev_data = data_to_vectors(read_file(dev_filename), glove)
    X_train, y_train = sep_X_y(train_data)
    X_dev, y_dev = sep_X_y(dev_data)

    #  svm model
    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)
    dev_svm_predictions = svm_model.predict(X_dev)
    print("F1 score for svm model is: {:.2f}".format(f1_score(y_dev, dev_svm_predictions)))

    # nn model
    net = Network()
    batch_size = 20
    train_data = TensorDataset(torch.FloatTensor(X_train),
                               torch.FloatTensor(y_train))
    dev_data = TensorDataset(torch.FloatTensor(X_dev),
                             torch.FloatTensor(y_dev))
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=batch_size)
    valid_loader = DataLoader(dev_data, shuffle=False,
                              batch_size=batch_size)
    epochs = 10
    print_every = 1000
    clip = 1000  # gradient clipping
    net.train()
    net = train_nn(net, epochs, print_every)
    X_dev_tensor = torch.FloatTensor(X_dev).unsqueeze(0)
    X_dev_tensor = X_dev_tensor.view(len(X_dev), -1)
    nn_dev_predictions = torch.round(net(X_dev_tensor)).detach().numpy()
    print("F1 score for nn model is: {:.2f}".format(
        f1_score(y_dev, nn_dev_predictions)))






