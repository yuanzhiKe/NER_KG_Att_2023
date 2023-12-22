import sys

sys.path.append("..")

from configs.CMeEE_config import (
    DEVICE,
    OUTPUT_DIR,
)
from datetime import datetime
from .trainer import NCBITrainer

class CBLUETrainer(NCBITrainer):
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        task_name,
        my_model_name,
        is_roberta=False,
        output_dir=OUTPUT_DIR
    ):
        super().__init__(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            task_name,
            my_model_name,
            is_roberta,
            output_dir=output_dir
        )

    def unpack_data_item(self, item):
        input_ids = item[0].to(DEVICE)

        if not self.is_roberta:
            token_type_ids = item[1].to(DEVICE)
            attention_mask = item[2].to(DEVICE)
            labels = item[3].to(DEVICE)
        else:
            attention_mask = item[2].to(DEVICE)
            labels = item[3].to(DEVICE)
            token_type_ids = None

        return input_ids, token_type_ids, attention_mask, labels