'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from utils import binary_sampler
from tensorflow.keras.datasets import mnist
import pandas as pd
import os.path
from os import path


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  usecols = 0
  # Load data
  if data_name in ['letter', 'spam']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  elif data_name == 'breast_original':
    file_name = 'data/' + data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", usecols=(range(30)), skiprows=1)
    data_y = np.loadtxt(file_name, delimiter=",", usecols=(30), skiprows=1)
    usecols = range(30)
    #print(data_x.shape)
    #print(data_y.shape)
  elif data_name == 'Wine_original':
    file_name = 'data/' + data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", usecols=(range(12)), skiprows=1)
    data_y = np.loadtxt(file_name, delimiter=",", usecols=(13), skiprows=1)
    usecols = range(12)
  elif data_name == 'mnist':
    (data_x, data_y), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
    data_y = np.reshape(np.asarray(data_y), [60000, 1]).astype(float)
  elif data_name == 'vals_test_df':
    train_data_name = "vals_train_df.csv"
    file_name = data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", usecols=(range(1,10)), skiprows=1)
    data_y = np.loadtxt(file_name, delimiter=",", usecols=(0), skiprows=1)
  elif data_name == 'vals_test_df_test_type1':
    train_data_name = "vals_train_df_test_type1.csv"
    file_name = data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", usecols=(range(1,10)), skiprows=1)
    data_y = np.loadtxt(file_name, delimiter=",", usecols=(0), skiprows=1)
  elif data_name == 'vals_test_df_test_type2':
    train_data_name = "vals_train_df_test_type2.csv"
    file_name = data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", usecols=(range(1, 10)), skiprows=1)
    data_y = np.loadtxt(file_name, delimiter=",", usecols=(0), skiprows=1)
  # Parameters
  no, dim = data_x.shape
  print(data_x.shape)


  # Introduce missing data
  #create missing file name
  filename = "{dname}_Missing_rate_{value}_Index.csv".format(dname=data_name, value = miss_rate)
  missing_file_exist = path.exists(filename)
  if missing_file_exist:
    file = open(filename)
    data_m = np.loadtxt(file, delimiter=",", usecols=(range(9)), skiprows=1)
    #print(data_m.shape)
  else:
    data_m = binary_sampler(1 - miss_rate, no, dim)
    #print(data_m.shape)
  #print("datax", data_x.shape)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
  #data_m = binary_sampler(1 - miss_rate, no, dim)
  data_train_x = np.loadtxt(train_data_name, delimiter=",", usecols=(range(1, 10)), skiprows=1)
  data_train_y = np.loadtxt(train_data_name, delimiter=",", usecols=(0), skiprows=1)
  miss_data_x_new = np.concatenate([data_train_x, miss_data_x])


  ### Saving the indexs
  missing_index = pd.DataFrame(data_m)
  missing_index.to_csv(filename, index=False)

  data_x_x = pd.DataFrame(miss_data_x)
  data_y_y = pd.DataFrame(data_y)
  data_x_s = pd.concat([data_y_y, data_x_x], ignore_index=True, axis=1)
  data_x_s.to_csv('{dbname}_generated.csv'.format(dbname=data_name), index=False)

  # print("data_x", data_x.shape)
  # print("miss_data_x_new", miss_data_x_new.shape)
  # print("data_m", data_m.shape)
  # print("data_y", data_y.shape)


  return data_x, miss_data_x_new, data_m, data_y