from hw1_help_funcs import *

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

    # GLOVE_PATH = 'glove-twitter-200' # ToDo - return to code
    # glove = downloader.load(GLOVE_PATH)

    train_data = data_to_vectors(read_file(train_filename), glove)
    dev_data = data_to_vectors(read_file(dev_filename), glove)
    X_train, y_train = sep_X_y(train_data)
    X_dev, y_dev = sep_X_y(dev_data)
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
    epochs = 100
    print_every = 1000
    clip = 1000  # gradient clipping
    net.train()
    net = train_nn(net, train_loader, valid_loader, clip, epochs, print_every)
    X_dev_tensor = torch.FloatTensor(X_dev).unsqueeze(0)
    X_dev_tensor = X_dev_tensor.view(len(X_dev), -1)
    nn_dev_predictions = torch.round(net(X_dev_tensor)).detach().numpy()
    print("F1 score for nn model is: {:.2f}".format(
        f1_score(y_dev, nn_dev_predictions)))
