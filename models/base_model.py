import torch
import logging
import os

from torch import nn
from transformers import AutoConfig, AutoModel


class NER_Model(nn.Module):
    def __init__(self, model_name, num_labels, my_model_name):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.my_model_name = my_model_name

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output = output.last_hidden_state
        output = self.dropout(output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:  # 输入正确答案标签时
            loss_fct = nn.CrossEntropyLoss()  # 交叉熵误差
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

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
