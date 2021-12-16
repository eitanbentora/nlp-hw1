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
import copy
from pathlib import Path


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
    mode = path.split('.')[-1]
    data = []
    with open(path, 'r', encoding="utf8") as file:
        current_sentence = []
        for line in file:
            if line != '\n':
                current_word = line.replace('\n', '').split("\t")
                if mode == 'tagged':
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


def get_additional_features(word, signs):
    """features to add to vector representation"""
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
    # firs letter is capital
    to_add.append(float(word[0][0].lower() != word[0]))
    # has 2 non sequential capital
    non_seq_cap = 0
    if len(word[0]) > 2:
        if word[0][0] != word[0][0].lower() and word[0][1] == word[0][1].lower():
            if word[0][2:].lower() != word[0][1:].lower():
                non_seq_cap = 1
    to_add.append(non_seq_cap)
    # has @
    has_at = '@' in word[0]
    to_add.append(float(has_at))
    has_hash = '#' in word[0]
    to_add.append(float(has_hash))

    # starts, endings and actions
    to_add.append(float(word[0].lower() in signs))
    to_add.append(float(word[0].lower() in ['won', 'went', 'did']))
    to_add.append(float(word[0].lower() in ['at', 'on', 'in', 'to']))
    prefixes = ['re']
    suffixes = ['ed', 'er', 'ing']
    for prefix in prefixes:
        to_add.append(float(word[0].lower().startswith(prefix)))
    for suffix in suffixes:
        to_add.append(float(word[0].lower().endswith(suffix)))
    return to_add


def replace_word(word, glove, signs):
    # has num
    for let in word[0]:
        if ord('9') >= ord(let) >= ord('0'):
            word[0] = 'seven'
            break
    only_signs = True
    for let in word[0]:
        if let not in signs:
            only_signs = False
            break
    if only_signs:
        word[0] = word[0][0]

    replace_signs = ["'", '#', '*', '.', '[', ']', '(', ')', '-']
    if len(word[0]) > 1:
        for sign in replace_signs:
            if word[0].lower().replace(sign, '') in glove.key_to_index:
                word[0] = word[0].replace(sign, '')
    if len(word[0]) == 2:
        if word[0][0] == ':' or word[0][0] == '=' or word[0][0] == ';':
            if word[0][1] == 'D':
                word[0] = 'smiley'
    return word


def data_to_vectors_extra_features(data, glove, word2vec_model=None):
    """
    :param data: data of words created by read_file
    :return: data with vectorized glove representation for the words.
    if not in glove, use the the vector found in the word2vec model trained
    """
    vec_data = copy.deepcopy(data)
    sentences = [[word[0].lower() for word in sen] for sen in data]
    signs = ['@', '#', '$', '%', '^', '&', '*', '(', ')', ':', '}', '{', ';', '.', '/', ',', '?', '~', '!', '[', ']',
             '-', '_', '"']
    if word2vec_model is None:
        word2vec_model = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=1, workers=4, epochs=100)
    missing_glove_words = 0
    for i, sen in enumerate(vec_data):
        for j, word in enumerate(sen):
            # add additional data
            to_add = get_additional_features(word, signs)
            # replace word if it is not in glove
            word = replace_word(word, glove, signs)
            if word[0].lower() not in glove.key_to_index:
                if word[0].lower() not in word2vec_model.wv:
                    missing_glove_words += 1
                    # TODO instead of this use the sentence's/window's avg
                    # print("not suppose to happen...")  # TODO remove
                    word[0] = (np.random.random(200) * 2 - 1)
                else:
                    word[0] = word2vec_model.wv[word[0].lower()]
            else:
                word[0] = glove[word[0].lower()]
            word[0] = np.concatenate((word[0], to_add))
    if missing_glove_words > 0:
        print("Amount of words missing from glove: ", missing_glove_words)
    return vec_data, word2vec_model


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
        x = self.layer_1(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.activation(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.layer_2(x)  # x.size() -> [batch_size, 1]
        x = torch.sigmoid(x)
        return x


def train_nn(net, train_loader, valid_loader, weights=(1, 1), clip=1000
             , epochs=10, print_every=1000, lr=0.0002, optimizer='Adam', loss_func='BCELoss'):
    net.train()
    optimizer_dict = {'Adam': torch.optim.Adam(net.parameters(), lr=lr),
                      'SGD': torch.optim.SGD(net.parameters(), lr=lr)}
    loss_func_dict = {'BCELoss': torch.nn.BCELoss(reduction='none'),
                      'MSELoss': torch.nn.MSELoss(reduction='none')}
    optimizer = optimizer_dict[optimizer]
    loss_func = loss_func_dict[loss_func]
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
            weights = torch.from_numpy(
                np.array([1 if i == 1 else 0.7 for i in labels]))
            temp_loss = loss_func(predictions.squeeze(),
                                  labels.float().squeeze())
            loss = torch.mean(weights * temp_loss)
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
                    val_weights = torch.from_numpy(
                        np.array([1 if i == 1 else 0.7 for i in labels]))
                    temp_val_loss = loss_func(val_predictions.squeeze(),
                                              labels.float().squeeze())
                    val_loss = torch.mean(val_weights * temp_val_loss)
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
    return net


class Network2(nn.Module):
    def __init__(self, input_dim=600):
        super().__init__()
        self.hidden_dim1 = 100
        self.hidden_dim2 = 10
        self.layer_1 = torch.nn.Linear(input_dim, self.hidden_dim1)
        self.layer_2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.layer_3 = torch.nn.Linear(self.hidden_dim2, 1)
        self.activation = F.relu

    def forward(self, x):
        x = self.layer_1(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.activation(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.layer_2(x)  # x.size() -> [batch_size, 1]
        x = self.activation(x)
        x = self.layer_3(x)
        x = torch.sigmoid(x)
        return x


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
    one_vec_data = np.zeros(length * vec_length)
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
                    non_data_idx.append(index + jump)
                    np.put(one_vec_data, np.arange(vec_length * (index + jump),
                                                   vec_length * (index + jump) + vec_length), sen_separator)
                    jump += 1

            np.put(one_vec_data, np.arange(vec_length * (index + jump), vec_length * (index + jump) + vec_length),
                   word[0])
            index += 1
    # print(f'jump is {jump}')
    non_data_add = [length - k for k in range(context_size, 0, -1)]
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
            x_mean_start += x[i * vec_length: (i + 1) * vec_length]
            x_mean_end += x[len(x) - (i + 1) * vec_length: len(x) - i * vec_length]
        x_mean_start, x_mean_end = x_mean_start / context_size, x_mean_end / context_size
        X_new.append(
            np.concatenate((x_mean_start, x[context_size * vec_length: (context_size + 1) * vec_length], x_mean_end)))
    return X_new


def prepare_data(train_filename, dev_filename, test_filename, context_size):
    # get glove model
    if Path('glove.pickle').is_file():
        with open('glove.pickle', 'rb') as handle:
            glove = pickle.load(handle)
    else:
        GLOVE_PATH = 'glove-twitter-200'
        glove = downloader.load(GLOVE_PATH)
    # read data files
    train_data = read_file(train_filename)
    dev_data = read_file(dev_filename)
    test_data = read_file(test_filename)
    # get data embedding
    vec_train, word2vec_model = data_to_vectors_extra_features(train_data, glove)
    vec_dev, _ = data_to_vectors_extra_features(dev_data, glove, word2vec_model)
    vec_test, _ = data_to_vectors_extra_features(test_data, glove, word2vec_model)
    # get labels
    _, y_train = sep_X_y(vec_train)
    _, y_dev = sep_X_y(vec_dev)

    vec_length = vec_train[0][0][0].shape[0]

    sen_separator = np.random.uniform(-1, 1, vec_length)
    # add context
    X_train = add_context(vec_train, context_size, sen_separator=sen_separator)
    X_dev = add_context(vec_dev, context_size, sen_separator=sen_separator)
    X_test = add_context(vec_test, context_size, sen_separator=sen_separator)
    # aggregate the context
    X_train_agg = agg_context_mean(X_train, context_size, vec_length)
    X_dev_agg = agg_context_mean(X_dev, context_size, vec_length)
    X_test_agg = agg_context_mean(X_test, context_size, vec_length)

    return X_train_agg, y_train, X_dev_agg, y_dev, X_test_agg

def write_predictions(pred, untagged_path, tagged_path):
    with open(untagged_path, 'r', encoding="utf8") as file:
        labeled_file = ''
        word_index = 0
        for line in file:
            if line == '\n':
                current_word = '\n'
            else:
                word_label = pred[word_index][0]
                current_word = line.replace('\n', '').split("\t")[0] + '\t' + str(word_label) + '\n'
                word_index += 1
            labeled_file += current_word
    if len(labeled_file) != word_index:
        print("In write predictions, last tag was", word_index + 1, "and the predictions length is", len(pred))
    with open(tagged_path, 'w', encoding="utf8") as file:
        file.write(labeled_file)




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


