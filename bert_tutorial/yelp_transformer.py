import torch
from transformers import BertTokenizer
from transformers.tokenization_utils import BatchEncoding


class YelpTransformer:
    def __init__(self, tokenizer: BertTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def transform(self, review: str):
        tokenized_review = self.tokenizer.tokenize(review)
        returned_dict: BatchEncoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            pad_to_max_length=True,
            is_pretokenized=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        encoded_reviews, token_type_ids, attention_masks = returned_dict.values()

        # tensor に変換
        tokens_tensor = torch.LongTensor(encoded_reviews).to('cpu')
        token_type_ids_tensor = torch.LongTensor(token_type_ids).to('cpu')
        attention_tensor = torch.FloatTensor(attention_masks).to('cpu')

        return tokens_tensor, token_type_ids_tensor, attention_tensor
