import time
import sys
import os
import pandas
import torch
import torch.optim as optim
from os import path
from torch import cuda
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from tqdm import tqdm
from yelp_dataset import YelpDataset
from yelp_transformer import YelpTransformer
from bert_model import BertMlp

N_STARS = 'n_stars'
REVIEWS = 'reviews'

# Settings
current_file_path = f'{os.getcwd()}/{ __file__}'
src_dir = path.dirname(current_file_path)
data_dir = f'{src_dir}/data'
train_path = f'{data_dir}/train.csv'
test_path = f'{data_dir}/test.csv'
model_path = f'{data_dir}/model.pth'

column_names = [N_STARS, REVIEWS]
num_data = 5000
batch_size = 128
epochs = 3

device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Use {device.upper()}.')

if not (path.exists(train_path) and path.exists(test_path)):
    print('File not found.')
    sys.exit(1)


def decrease(label): return label - 1


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

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

batch_iterator = iter(train_dataloader)
encoded_reviews, token_type_ids, attention_mask, label_tensor = next(
    batch_iterator)

print(encoded_reviews.shape)


model = BertMlp('bert-base-uncased').to(device)

crietion = CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# model.to(device)

model.train()

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-'*10)

    epoch_loss = 0.0
    epoch_corrects = 0

    for batch, (encoded_tokens, token_type_ids, attention_mask, labels) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(encoded_tokens, token_type_ids, attention_mask)

        loss = crietion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs, 1)
        epoch_loss += loss.item()
        epoch_corrects += torch.sum(pred == labels)

    dataset_len = len(train_dataloader.dataset)
    epoch_loss = epoch_loss / len(dataset_len)
    epoch_acc = epoch_corrects.double() / len(dataset_len)
    print(f'\n{"="*10}')
    print('Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))
    print('='*10)

checkpoints = {
    'model_state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'criterion': crietion,
}

print('saving at %s' % (model_path))
torch.save(checkpoints, model_path)
