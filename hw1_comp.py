from hw1_help_funcs import *
import pickle

if __name__ == '__main__':
    set_seed()
    to_load_processed_data = False

    _, train_filename, dev_filename, test_filename = sys.argv
    print(train_filename, dev_filename, test_filename)
    context_size = 4
    # load or create processed_data
    if Path('processed_data.pkl').is_file() and to_load_processed_data:
        with open('processed_data.pkl', 'rb') as handle:
            X_train, y_train, X_dev, y_dev, X_test = pickle.load(handle)
    else:
        X_train, y_train, X_dev, y_dev, X_test = prepare_data(train_filename, dev_filename, test_filename, context_size)
        with open('processed_data.pkl', 'wb') as handle:
            pickle.dump([X_train, y_train, X_dev, y_dev, X_test], handle, protocol=pickle.HIGHEST_PROTOCOL)

    input_dim = len(X_train[0])
    print('The size of the embedded word vector is', input_dim)
    net = Network2(input_dim=input_dim)

    # create Tensor datasets
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    dev_data = TensorDataset(torch.FloatTensor(X_dev), torch.FloatTensor(y_dev))

    # make sure the SHUFFLE your training data
    batch_size = 20

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)

    net_params = {'epochs': 6, 'print_every': 1000, 'clip': 1000,
                  'lr': 0.0002, 'optimizer': 'Adam',
                  'loss_func': 'BCELoss', 'weights': [0.7, 1]}

    net = train_nn(net, train_loader, valid_loader, **net_params)

    X_dev_tensor = torch.FloatTensor(X_dev).unsqueeze(0)
    X_dev_tensor = X_dev_tensor.view(len(X_dev), -1)

    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(0)
    X_test_tensor = X_test_tensor.view(len(X_test), -1)

    nn_dev_pred = torch.round(net(X_dev_tensor)).detach().numpy()
    print("F1 score for nn model is: {:.3f}".format(f1_score(y_dev, nn_dev_pred)))

    nn_test_pred = torch.round(net(X_test_tensor)).detach().numpy()

    write_predictions(nn_test_pred, '../data/test.untagged', '../data/test_comp.tagged')

