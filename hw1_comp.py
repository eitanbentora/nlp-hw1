from hw1_help_funcs import *

if __name__ == '__main__':
    set_seed()
    ####################### To Remove
    with open('glove.pickle', 'rb') as handle:
        glove = pickle.load(handle)
    #######################

    _, train_filename, dev_filename, test_filename = sys.argv
    # train_filename, = r'..\data\train.tagged'
    # dev_filename = r'..\data\dev.tagged'
    print(train_filename, dev_filename, test_filename)

    # GLOVE_PATH = 'glove-twitter-200' # ToDo - return to code
    # glove = downloader.load(GLOVE_PATH)

    train_data = read_file(train_filename)
    dev_data = read_file(dev_filename)
    test_data = read_file(test_filename)

    vec_train, word2vec_model = data_to_vectors_extra_features(train_data, glove)
    vec_dev, _ = data_to_vectors_extra_features(dev_data, glove, word2vec_model)
    vec_test, _ = data_to_vectors_extra_features(test_data, glove, word2vec_model)

    _, y_train = sep_X_y(vec_train)
    _, y_dev = sep_X_y(vec_dev)


    batch_size = 20
    context_size = 4
    set_seed()

    vec_length = vec_train[0][0][0].shape[0]

    sen_separator = np.random.uniform(-1, 1, vec_length)

    X_train = add_context(vec_train, context_size, sen_separator=sen_separator)
    X_dev = add_context(vec_dev, context_size, sen_separator=sen_separator)
    X_test = add_context(vec_test, context_size, sen_separator=sen_separator)

    X_train_agg = agg_context_mean(X_train, context_size, vec_length)
    X_dev_agg = agg_context_mean(X_dev, context_size, vec_length)
    X_test_agg = agg_context_mean(X_test, context_size, vec_length)

    input_dim = len(X_train_agg[0])
    print('The size of the embedded word vector is', input_dim)

    net = Network2(input_dim=input_dim)

    # create Tensor datasets
    train_data = TensorDataset(torch.FloatTensor(X_train_agg), torch.FloatTensor(y_train))
    dev_data = TensorDataset(torch.FloatTensor(X_dev_agg), torch.FloatTensor(y_dev))

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)

    net_params = {'epochs': 6, 'print_every': 1000, 'clip': 1000,
                  'lr': 0.0002, 'optimizer': 'Adam',
                  'loss_func': 'BCELoss', 'weights': [0.7, 1]}

    net = train_nn(net, train_loader, valid_loader, **net_params)

    X_dev_tensor = torch.FloatTensor(X_dev_agg).unsqueeze(0)
    X_dev_tensor = X_dev_tensor.view(len(X_dev_agg), -1)

    X_test_tensor = torch.FloatTensor(X_test_agg).unsqueeze(0)
    X_test_tensor = X_dev_tensor.view(len(X_test_agg), -1)

    nn_dev_predictions = torch.round(net(X_dev_tensor)).detach().numpy()
    print("F1 score for nn model is: {:.3f}".format(f1_score(y_dev, nn_dev_predictions)))
