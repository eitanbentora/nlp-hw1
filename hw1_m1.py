from hw1_help_funcs import *
import time
import pickle

if __name__ == '__main__':

    set_seed()
    _, train_filename, dev_filename, test_filename, save_output_path = sys.argv
    print(train_filename, dev_filename, test_filename, save_output_path)

    to_load_processed_data = False

    context_size = 4
    # load or create processed_data
    if Path('processed_data.pkl').is_file() and to_load_processed_data:
        with open('processed_data.pkl', 'rb') as handle:
            X_train, y_train, X_dev, y_dev, X_test = pickle.load(handle)
    else:
        X_train, y_train, X_dev, y_dev, X_test = prepare_data(train_filename, dev_filename, test_filename, context_size)
        with open('processed_data.pkl', 'wb') as handle:
            pickle.dump([X_train, y_train, X_dev, y_dev, X_test], handle, protocol=pickle.HIGHEST_PROTOCOL)


    #  svm model
    best_f1 = 0
    best_C = None
    best_model = None
    for C in [3.2]:
        start = time.time()
        svm_model = svm.SVC(C=C)
        svm_model.fit(X_train, y_train)
        with open("models/m1.pkl", 'wb') as f:
            pickle.dump(svm_model, f)
        dev_svm_predictions = svm_model.predict(X_dev)
        f1 = f1_score(y_dev, dev_svm_predictions)
        if f1 > best_f1:
            best_f1, best_C, best_model = f1, C, svm_model
        print("F1 score for svm model is: {:.5f}".format(f1), f"for C={C}")
        print(f"time: {time.time()-start}")
    print(f"best C is {best_C}, with F1 score {np.round(best_f1, 3)}")
    test_svm_predictions = best_model.predict(X_test)
    write_predictions([[pred] for pred in test_svm_predictions], test_filename, save_output_path)
