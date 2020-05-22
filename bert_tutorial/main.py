import time
import sys
import os
import pandas
import torch
from os import path
from transformers import BertTokenizer
from .yelp_dataset import YelpDataset
from .yelp_transformer import YelpTransformer
from torch.utils.data.dataloader import DataLoader

N_STARS = 'n_stars'
REVIEWS = 'reviews'

## Settings
current_file_path = f'{os.getcwd()}/{ __file__}'
src_dir = path.dirname(current_file_path)
data_dir = f'{src_dir}/data'
train_path = f'{data_dir}/train.csv'
test_path = f'{data_dir}/test.csv'

column_names = [N_STARS, REVIEWS]
num_data = 5000
batch_size = 128

if not (path.exists(train_path) and path.exists(test_path)):
  print('File not found.')
  sys.exit(1)

decrease = lambda label: label - 1
train_data_frame: pandas.DataFrame = pandas.read_csv(train_path, names=column_names) \
  .replace(to_replace='\n', value=' ')
train_data_frame[N_STARS] = train_data_frame[N_STARS].map(decrease)
test_data_frame: pandas.DataFrame = pandas.read_csv(test_path, names=column_names) \
  .replace(to_replace='\n', value=' ')
test_data_frame[N_STARS] = test_data_frame[N_STARS].map(decrease)

train_review = train_data_frame[REVIEWS][:num_data]
train_labels = train_data_frame[N_STARS][:num_data]
test_review = train_data_frame[REVIEWS][:num_data]
test_labels = train_data_frame[N_STARS][:num_data]

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
transformer = YelpTransformer(bert_tokenizer)
train_dataset = YelpDataset(
  reviews=train_review,
  labels=train_labels,
  transformer=transformer,
)
test_dataset = YelpDataset(
  reviews=test_review,
  labels=test_labels,
  transformer=transformer,
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)