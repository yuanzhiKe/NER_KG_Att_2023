import logging
import numpy as np
from torch.utils.data import Dataset


class NCBIDataset(Dataset):
    def __init__(self, ncbi_data, tokenizer, max_len, is_roberta=False):
        # ncbi_data: dataset["train"] or dataset["validation"] or dataset["test"]
        super().__init__()
        self.is_roberta = is_roberta
        self.data_ids = []
        self.input_ids = []
        if not self.is_roberta:
            self.token_type_ids = []
        self.attention_mask = []
        self.tags = []
        self.maxlen = max_len
        self.tokenizer = tokenizer
        self.preprocess(ncbi_data)

    def preprocess(self, ncbi_data):
        for data in ncbi_data:
            self.data_ids.append(data["id"])
            input_ids = [101]  # start with <cls>
            if not self.is_roberta:
                token_type_ids = [0]
            attention_mask = [1]
            tags = [-100]  # start with pad
            for token, tag in zip(data["tokens"], data["ner_tags"]):
                results = self.tokenizer.encode_plus(token)
                local_input_ids = results["input_ids"]
                if not self.is_roberta:
                    local_token_type_ids = results["token_type_ids"]
                local_attention_mask = results["attention_mask"]
                local_tags = [tag] * len(local_input_ids)
                input_ids += local_input_ids
                if not self.is_roberta:
                    token_type_ids += local_token_type_ids
                attention_mask += local_attention_mask
                tags += local_tags
            if len(input_ids) > self.maxlen:
                input_ids = input_ids[: self.maxlen - 1]
                if not self.is_roberta:
                    token_type_ids = token_type_ids[: self.maxlen - 1]
                attention_mask = attention_mask[: self.maxlen - 1]
                tags = tags[: self.maxlen - 1]
                input_ids += [102]  # end with <sep>
                if not self.is_roberta:
                    token_type_ids += [0]
                attention_mask += [1]
                tags += [-100]
            else:
                padding_len = self.maxlen - len(input_ids) - 1
                input_ids += [102] + [0] * padding_len
                if not self.is_roberta:
                    token_type_ids += [0] + [0] * padding_len
                attention_mask += [1] + [0] * padding_len
                tags += [-100] + [-100] * padding_len
            self.input_ids.append(input_ids)
            if not self.is_roberta:
                self.token_type_ids.append(token_type_ids)
            self.attention_mask.append(attention_mask)
            self.tags.append(tags)
        self.input_ids = np.array(self.input_ids, dtype=np.int32)
        if not self.is_roberta:
            self.token_type_ids = np.array(self.token_type_ids, dtype=np.int32)
        self.attention_mask = np.array(self.attention_mask, dtype=np.int32)
        self.tags = np.array(self.tags)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        if not self.is_roberta:
            return (
                self.input_ids[index],
                self.token_type_ids[index],
                self.attention_mask[index],
                self.tags[index],
            )
        else:
            return (
                self.input_ids[index],
                self.attention_mask[index],
                self.tags[index],
            )
