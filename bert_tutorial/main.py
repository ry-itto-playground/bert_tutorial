import sys
import os
import pandas
from os import path

# This filepath
current_file_path = f'{os.getcwd()}/{ __file__}'
src_dir = path.dirname(current_file_path)
data_dir = f'{src_dir}/data'

train_path = f'{data_dir}/train.csv'
test_path = f'{data_dir}/test.csv'

if not (path.exists(train_path) and path.exists(test_path)):
  print('File not found.')
  sys.exit(1)

names = ["n_stars", "review"]

train_data_frame: pandas.DataFrame = pandas.read_csv(train_path, names=names) \
  .replace(to_replace='\n', value=' ')
train_data_frame['n_stars'].apply(lambda label: label - 1)
test_data_frame: pandas.DataFrame = pandas.read_csv(test_path, names=names) \
  .replace(to_replace='\n', value=' ')

print(train_data_frame.head())