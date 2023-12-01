import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/CBLUE/")

import numpy as np
from configs.CMeEE_config import MAX_LEN

from cblue.data import EEDataProcessor
from cblue.data import EEDataset

class MyEEDataProcessor(EEDataProcessor):
  def __init__(self, root, is_lower=True, no_entity_label='O'):
    super().__init__(root, is_lower, no_entity_label)
    self.task_data_dir = root
    self.train_path = os.path.join(self.task_data_dir, 'CMeEE_train.json')
    self.dev_path = os.path.join(self.task_data_dir, 'CMeEE_dev.json')
    self.test_path = os.path.join(self.task_data_dir, 'CMeEE_test.json')
    self.label_map_cache_path = os.path.join(self.task_data_dir, 'CMeEE_label_map.dict')

class MyEEDataset(EEDataset):
  def __init__(
    self,
    data_path,
    tokenizer,
    type="train",
    mode='train',
    max_length=MAX_LEN,
    ignore_label=-100,
    model_type='bert',
    ngram_dict=None
  ):
    assert type in ["train", "eval", "test"]
    data_processor =  MyEEDataProcessor(data_path)
    if type == "train":
      samples = data_processor.get_train_sample()
    elif type == "eval":
      samples = data_processor.get_eval_sample()
    elif type == "test":
      samples = data_processor.get_test_sample()
    else:
      raise TypeError("the type argument should be chosen from 'train', 'eval', or 'test'.")
    super(MyEEDataset, self).__init__(
      samples,
      data_processor,
      tokenizer,
      mode,
      max_length,
      ignore_label,
      model_type,
      ngram_dict
    )

  def __getitem__(self, idx):
    text = self.orig_text[idx]
    inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
    if self.mode != "test":
      label = [self.data_processor.label2id[label_] for label_ in
            self.labels[idx].split('\002')]  # find index from label list
      label = ([-100] + label[:self.max_length - 2] + [-100] +
            [self.ignore_label] * self.max_length)[:self.max_length]  # use ignore_label padding CLS+label+SEP
      return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
          np.array(inputs['attention_mask']), np.array(label)
    else:
      return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
          np.array(inputs['attention_mask']),

  def get_data_processor(self):
    return self.data_processor

def test():
  from transformers import AutoTokenizer
  train = MyEEDataset(os.path.dirname(os.path.realpath(__file__)) + "/../CMeEE", tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese"), type="train", mode="train")
  eval = MyEEDataset(os.path.dirname(os.path.realpath(__file__)) + "/../CMeEE", tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese"), type="eval", mode="train")
  test = MyEEDataset(os.path.dirname(os.path.realpath(__file__)) + "/../CMeEE", tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese"), type="test", mode="test")
  assert(train)
  assert(eval)
  assert(test)

if __name__ == "main":
  test()