import sys
import os

import numpy as np
from configs.CMeEE_config import MAX_LEN

from cblue.data import EEDataProcessor
from cblue.data import EEDataset

class MyEEDataProcessor(EEDataProcessor):
  def __init__(self, root, is_lower=True, no_entity_label='O'):
    # super().__init__(root, is_lower, no_entity_label)
    self.task_data_dir = os.path.join(os.path.dirname(__file__), "..", root)
    self.train_path = os.path.join(self.task_data_dir, 'CMeEE-V2_train.json')
    self.dev_path = os.path.join(self.task_data_dir, 'CMeEE-V2_dev.json')
    self.test_path = os.path.join(self.task_data_dir, 'CMeEE-V2_test.json')

    self.label_map_cache_path = os.path.join(self.task_data_dir, 'CMeEE_label_map.dict')
    self.label2id = None
    self.id2label = None
    self.no_entity_label = no_entity_label
    self._get_labels()
    self.num_labels = len(self.label2id.keys())

    self.is_lower = is_lower
  

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
      samples = data_processor.get_dev_sample()
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
    self.preprocess()

  def preprocess(self):
    input_ids, token_type_ids, attention_masks, processed_labels = [], [], [], []
    for i in range(len(self.orig_text)):
        text = self.orig_text[i]
        inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
        if self.mode != "test":
          label = [self.data_processor.label2id[label_] for label_ in
            self.labels[i].split('\002')]  # find index from label list
          label = ([-100] + label[:self.max_length - 2] + [-100] +
              [self.ignore_label] * self.max_length)[:self.max_length]  # use ignore_label padding CLS+label+SEP
          input_ids.append(inputs['input_ids'])
          token_type_ids.append(inputs['token_type_ids'])
          attention_masks.append(inputs['attention_mask'])
          processed_labels.append(label)
        else:
          input_ids.append(inputs['input_ids'])
          token_type_ids.append(inputs['token_type_ids'])
          attention_masks.append(inputs['attention_mask'])
    self.input_ids = input_ids
    self.token_type_ids = token_type_ids
    self.attention_masks = attention_masks
    self.labels = processed_labels

  def __getitem__(self, idx):
    if self.mode != "test":
      return np.array(self.input_ids[idx]), np.array(self.token_type_ids[idx]), \
          np.array(self.attention_masks[idx]), np.array(self.labels[idx])
    else:
      return np.array(self.input_ids[idx]), np.array(self.token_type_ids[idx]), \
          np.array(self.attention_masks[idx])

  def get_data_processor(self):
    return self.data_processor
