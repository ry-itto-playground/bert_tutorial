from transformers import BertTokenizer, PreTrainedTokenizer

# Use pretrained model
bert_tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained('bert-base-cased')

### Tokenize example
test_sentence = 'i love cupcakes'
test_tokenized = bert_tokenizer.tokenize(test_sentence)

returned_dict = bert_tokenizer.encode_plus(
  test_tokenized,
  add_special_tokens=True,
  max_length=bert_tokenizer.model_max_length,
  pad_to_max_length=True,
  is_pretokenized=True,
  return_token_type_ids=True,
  return_attention_mask=True,
  )
print(returned_dict.keys())

encoded_test, token_type_ids, attention_mask = returned_dict['input_ids'], returned_dict['token_type_ids'], returned_dict['attention_mask']
print("="*10)
print("original text: {}".format(test_sentence))
print("output type {}".format(type(encoded_test)))
print("length of output: {}".format(len(encoded_test)))
print("\nconverted to ids: {}".format(encoded_test[:10]))
print("token_type_ids: {}".format(token_type_ids[:10]))
print("attention_mask: {}".format(attention_mask[:10]))

print("\n 実際にトークナイズされたテキスト")
print("tokenized text: {}".format(bert_tokenizer.convert_ids_to_tokens(encoded_test[:10])))
print("="*10)
###