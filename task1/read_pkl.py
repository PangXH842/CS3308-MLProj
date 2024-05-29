import pickle
import numpy as np

with open("./task1/test_data/adder/adder_14.pkl", 'rb') as f:
    data = pickle.load(f)
    print(str(data))