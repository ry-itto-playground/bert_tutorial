import torch
from .yelp_transformer import YelpTransformer
from torch.utils.data.dataset import Dataset

class YelpDataset(Dataset):
  def __init__(self, reviews: list, labels: list, transformer: YelpTransformer):
    self.reviews=reviews
    self.labels=labels
    self.transform = transform

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, index: int) -> [torch.Tensor]:
    review = self.reviews[index]
    label = self.labels[index]

    encoded_reviews, token_type_ids, attention_mask = self.transformer.transform(review)
    label_tensor = torch.LongTensor(label).to('cpu')

    return [*self.transformer.transform(review), label_tensor]