import logging
import os
import torch

from torch import nn
from transformers import AutoConfig, AutoModel
from .cross_attention import CrossAttention


class NER_Model_SA(nn.Module):
    def __init__(self, model_name, num_labels, my_model_name):
        super().__init__()

        # 输入BERT参数
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # IOB2标签总数（B-人名，I-人名，B-地名，I-地名，…，O）
        self.num_labels = num_labels

        # 用于将BERT嵌入转换为IOB2标签的线性层
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.cross_att = CrossAttention(config.hidden_size)

        self.my_model_name = my_model_name

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output = output.last_hidden_state
        output = self.do_attention(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:  # 输入正确答案标签时
            loss_fct = nn.CrossEntropyLoss()  # 交叉熵误差
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def do_attention(self, input_vectors):
        # 重载该函数以实现使用了不同类型的attention的模型
        return self.cross_att(input_vectors, input_vectors)

    def save_pretrained(self, save_directory):
        """Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
        """
        if os.path.isfile(save_directory):
            logging.error(
                "Provided path ({}) should be a directory, not a file".format(
                    save_directory
                )
            )
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, self.my_model_name + ".bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logging.info("Model weights saved in {}".format(output_model_file))

    def load_pretrained(self, save_directory):
        output_model_file = os.path.join(save_directory, self.my_model_name + ".bin")
        best_state_dict = torch.load(output_model_file)
        if hasattr(self, "module"):
            self.module.load_state_dict(best_state_dict)
        else:
            self.load_state_dict(best_state_dict)
        return self


class NER_Model_OA(NER_Model_SA):
    def __init__(
        self,
        model_name,
        num_labels,
        my_model_name,
        ontology,
        tokenizer,
        device,
        is_roberta=False,
    ):
        super().__init__(model_name, num_labels, my_model_name)
        self.is_roberta = is_roberta
        self.set_ontology(ontology, tokenizer, device)

    def set_ontology(self, ontology, tokenizer, device):
        tokenzied_ontology = tokenizer.encode_plus(ontology, return_tensors="pt")
        self.ontology_input_ids = tokenzied_ontology["input_ids"].to(
            device
        )  # need to put into device here
        if not self.is_roberta:
            self.ontology_token_type_ids = tokenzied_ontology["token_type_ids"].to(
                device
            )
        self.ontology_attention_mask = tokenzied_ontology["attention_mask"].to(device)

    def encode_ontology(self, batch_len):
        ontology_input_ids = self.ontology_input_ids.repeat(batch_len, 1)
        ontology_attention_mask = self.ontology_attention_mask.repeat(batch_len, 1)
        if not self.is_roberta:
            ontology_token_type_ids = self.ontology_token_type_ids.repeat(batch_len, 1)
            return self.bert(
                ontology_input_ids,
                token_type_ids=ontology_token_type_ids,
                attention_mask=ontology_attention_mask,
            ).last_hidden_state
        else:
            return self.bert(
                ontology_input_ids,
                attention_mask=ontology_attention_mask,
            ).last_hidden_state

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output = output.last_hidden_state
        output = self.do_attention(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:  # 输入正确答案标签时
            loss_fct = nn.CrossEntropyLoss()  # 交叉熵误差
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def do_attention(self, input_vectors):
        ontology_vec = self.encode_ontology(input_vectors.shape[0])
        return self.cross_att(input_vectors, ontology_vec)
