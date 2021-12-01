import pandas as pd

filename = r'..\data\train.tagged'
train_data = pd.read_csv(filename, sep='\t', lineterminator='\r')
