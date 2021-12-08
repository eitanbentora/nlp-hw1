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
