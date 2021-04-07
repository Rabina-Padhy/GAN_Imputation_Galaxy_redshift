import numpy as np
import pandas as pd

data_train_x = np.loadtxt("vals_train_df.csv", delimiter=",", usecols=(range(1, 10)), skiprows=1)
data_train_y = np.loadtxt("vals_train_df.csv", delimiter=",", usecols=(0), skiprows=1)

print(data_train_x.dtype)
print(data_train_x.shape)

data_x = np.loadtxt("vals_test_df.csv", delimiter=",", usecols=(range(1, 10)), skiprows=1)
data_y = np.loadtxt("vals_test_df.csv", delimiter=",", usecols=(0), skiprows=1)

print(data_x.dtype, "dwdd1")

print(data_x.shape)

d = np.concatenate([data_train_x, data_x])
print(d.shape)

d1 = d[range(918,1311), :]
print(d1.shape)
print(d1)
