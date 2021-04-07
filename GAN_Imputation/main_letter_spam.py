'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import timeit
import pandas as pd

from data_loader import data_loader
from gain import gain
from utils import rmse_loss


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, data_y = data_loader(data_name, miss_rate)


  imputed_data_x = gain(miss_data_x, gain_parameters)



    #pd.DataFrame(data_y, imputed_data_x, axis = 1)

  # Step- craete data_m using testdata
  # Step - combine train and missing_test_data
  # Step - retrun total missing and original data_m
  # Step - while calculating RMSE
    # use original as test_original
    # fetch testing imputed datset 934 to last
    # data_m as missing_test_data

  if data_name == 'vals_test_df':
      imputed_data_x = imputed_data_x[range(918,1311), :]
  elif data_name == 'vals_test_df_test_type1':
      imputed_data_x = imputed_data_x[range(495, 1311), :]
  elif data_name == 'vals_test_df_test_type2':
      imputed_data_x = imputed_data_x[range(816, 1311), :]
  else:
      imputed_data_x = imputed_data_x



  imputed_data_x_df = pd.DataFrame(imputed_data_x)
  data_y_df = pd.DataFrame(data_y)
  imputed_data_df = pd.concat([data_y_df, imputed_data_x_df], ignore_index=True, axis=1)
  imputed_data_df.to_csv("GAN_imputated_catalogueData1.csv", index=False)

  # Report the RMSE performance
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)


  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'breast_original','Wine_original'],
      default='vals_test_df',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.15,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=60000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function with time recorded
#for i in range(11):
start = timeit.default_timer()
imputed_data, rmse = main(args)
stop = timeit.default_timer()
timeTaken = stop - start
print('Time to run the loop: ', timeTaken)

with open('out.txt', 'w') as f:
   print('RMSE:', rmse, file=f)  # Python 3.x
   print('Imputed data:', imputed_data, file=f)
   print('timeTaken:', timeTaken, file=f)


# which keep both the output

  #print("Iteration:", i)
