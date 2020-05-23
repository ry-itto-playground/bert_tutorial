import torch
from torch.nn.modules import Linear, Module
from transformers import BertModel


class BertMlp(Module):

    def __init__(self, bert_model_type: str):
        super(BertMlp, self).__init__()

        bert_output_size = 768

        self.bert_layer = BertModel.from_pretrained(bert_model_type)
        for param in self.bert_layer.parameters():
            param.requires_grad = False

        self.dence = Linear(bert_output_size, 5)

    def forward(self, tokens, token_type_ids, attention):
        # emb input (batch_size, seq_len)
        bert_embedded = self.bert_layer(
            input_ids=tokens,
            token_type_ids=token_type_ids,
            attention_mask=attention,
        )[0][:, 0, :]

        # emb output (batch_size, seq_len, bert_hidden_size)
        # bert_embedded = bert_embedded

        return self.dence(bert_embedded)
