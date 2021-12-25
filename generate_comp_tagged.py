import pickle
import sys

from hw1_help_funcs import *

if __name__ == '__main__':

    _, train_filename, dev_filename, test_filename = sys.argv
    print(train_filename, dev_filename, test_filename)
    to_load_processed_data = True

    context_size = 4
    # load or create processed_data
    if Path('processed_data.pkl').is_file() and to_load_processed_data:
        with open('processed_data.pkl', 'rb') as handle:
            X_train, y_train, X_dev, y_dev, X_test = pickle.load(handle)
    else:
        X_train, y_train, X_dev, y_dev, X_test = prepare_data(train_filename, dev_filename, test_filename, context_size)
        with open('processed_data.pkl', 'wb') as handle:
            pickle.dump([X_train, y_train, X_dev, y_dev, X_test], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("models/m1.pkl", 'rb') as f:
        m1_model = pickle.load(f)
    with open("models/m2.pkl", 'rb') as f:
        m2_model = pickle.load(f)
    with open("models/comp.pkl", 'rb') as f:
        comp_model = pickle.load(f)

    for model, model_name in zip([m1_model, m2_model, comp_model], ['m1', 'm2', 'comp']):
        print(model_name)
        save_output_path = f'test_{model_name}.tagged'
        if model_name == 'm2':
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(0)
            X_test_tensor = X_test_tensor.view(len(X_test), -1)
            predictions = torch.round(model(X_test_tensor)).detach().numpy()
            write_predictions(predictions, test_filename, save_output_path)

        else:
            predictions = model.predict(X_test)
            write_predictions([[pred] for pred in predictions], test_filename, save_output_path)

