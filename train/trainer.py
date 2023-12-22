import logging
import os
import sys
import torch
import numpy as np

sys.path.append("..")

from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from cblue.utils import ProgressBar, seed_everything
from sklearn.metrics import precision_recall_fscore_support
from configs.ncbi_config import (
    DEVICE,
    EPOCHS,
    WARMUP_PROPORTION,
    WEIGHT_DECAY,
    LEARNING_RATE,
    LOGGING_STEPS,
    ADAM_EPSILON,
    MAX_GRAD_NORM,
    EARLYSTOP_PATIENCE,
    OUTPUT_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    SEED,
)
from datetime import datetime

def macro_f1(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')

class NCBITrainer(object):
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        task_name,
        my_model_name,
        is_roberta=False,
        output_dir = OUTPUT_DIR
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_name = task_name
        self.my_model_name = my_model_name
        self.is_roberta = is_roberta
        self.output_dir = output_dir

    def train(self):
        model = self.model
        model.to(DEVICE)

        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * EPOCHS
        num_warmup_steps = num_training_steps * WARMUP_PROPORTION
        num_examples = len(train_dataloader.dataset)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if SEED is not None:
            seed_everything(SEED)
        model.zero_grad()

        logging.info("***** Running training *****")
        logging.info("Num samples %d", num_examples)
        logging.info("Num epochs %d", EPOCHS)
        logging.info("Num training steps %d", num_training_steps)
        logging.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = 0.0
        cnt_patience = 0
        for i in range(EPOCHS):
            logging.info("**Epoch %d", i)
            pbar = ProgressBar(n_total=len(train_dataloader), desc="Training")
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)

                if step % 10 == 0:
                    pbar(step, {"loss": loss.item()})

                if MAX_GRAD_NORM:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if LOGGING_STEPS > 0 and global_step % LOGGING_STEPS == 0:
                    print("")
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        logging.info(
                            "Earlystopper counter: %s out of %s",
                            cnt_patience,
                            EARLYSTOP_PATIENCE,
                        )
                        if cnt_patience >= EARLYSTOP_PATIENCE:
                            break
            if cnt_patience >= EARLYSTOP_PATIENCE:
                break

        logging.info("Training Stop! The best step %s: %s", best_step, best_score)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=best_step)

        return global_step, best_step

    def unpack_data_item(self, item):
        input_ids = item[0].to(DEVICE)

        if not self.is_roberta:
            token_type_ids = item[1].to(DEVICE)
            attention_mask = item[2].to(DEVICE)
            labels = item[3].to(DEVICE)
        else:
            attention_mask = item[1].to(DEVICE)
            labels = item[2].to(DEVICE)
            token_type_ids = None

        return input_ids, token_type_ids, attention_mask, labels

    def get_token(self, id):
        token = self.tokenizer.convert_ids_to_tokens(int(id))
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            return ""
        else:
            return token

    def evaluate(self, model, output_results=False):
        model.to(DEVICE)
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        if output_results:
            preds_for_output = None
            labels_for_output = None
            input_ids_for_output = None

        logging.info("***** Running evaluation *****")
        logging.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids, token_type_ids, attention_mask, labels = self.unpack_data_item(
                item
            )

            with torch.no_grad():
                outputs = model(
                    labels=labels,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )

                # outputs = model(labels=labels, **inputs)
                _, _logits = outputs[:2]
                if output_results:
                    active_logits_2d = _logits.argmax(dim=-1)
                    local_preds_2d = active_logits_2d.detach().cpu().numpy()
                    local_labels_2d = labels.detach().cpu().numpy()
                    local_input_ids = input_ids.detach().cpu().numpy()

                active_index = attention_mask.view(-1) == 1
                active_labels = labels.view(-1)[active_index]
                logits = _logits.argmax(dim=-1)
                active_logits = logits.view(-1)[active_index]
                local_preds = active_logits.detach().cpu().numpy()
                local_labels = active_labels.detach().cpu().numpy()

            if preds is None:
                preds = local_preds
                eval_labels = local_labels
                if output_results:
                    preds_for_output = local_preds_2d
                    labels_for_output = local_labels_2d
                    input_ids_for_output = local_input_ids
            else:
                preds = np.append(preds, local_preds, axis=0)
                eval_labels = np.append(eval_labels, local_labels, axis=0)
                if output_results:
                    preds_for_output = np.append(
                        preds_for_output, local_preds_2d, axis=0
                    )
                    labels_for_output = np.append(
                        labels_for_output, local_labels_2d, axis=0
                    )
                    input_ids_for_output = np.append(
                        input_ids_for_output, local_input_ids, axis=0
                    )

        if output_results:
            output_text = ""
            with open(
                os.path.join(
                    self.output_dir,
                    f"eval_outputs_{self.my_model_name}_{datetime.now():%Y%m%d%H%M%S}.txt",
                ),
                "w",
            ) as f:
                for i in range(preds_for_output.shape[0]):
                    output_text += f"ID: {self.eval_dataset.data_ids[i]} \t"
                    last_tag = 2
                    for j in range(preds_for_output.shape[1]):
                        if preds_for_output[i][j] == 1 and last_tag != 1:
                            token = self.get_token(int(input_ids_for_output[i][j]))
                            output_text += f"<S>{token}"
                            last_tag = preds_for_output[i][j]
                        elif preds_for_output[i][j] == 2 or preds_for_output[i][j] == 1:
                            token = self.get_token(int(input_ids_for_output[i][j]))
                            output_text += f"{token}"
                            last_tag = preds_for_output[i][j]
                    output_text += "\n"
                    output_text += "Gr T: \t"
                    last_tag = 2
                    for j in range(labels_for_output.shape[1]):
                        if labels_for_output[i][j] == 1 and last_tag != 1:
                            token = self.get_token(int(input_ids_for_output[i][j]))
                            output_text += f"<S>{token}"
                            last_tag = preds_for_output[i][j]
                        elif (
                            labels_for_output[i][j] == 2 or preds_for_output[i][j] == 1
                        ):
                            token = self.get_token(int(input_ids_for_output[i][j]))
                            output_text += f"{token}"
                            last_tag = preds_for_output[i][j]
                    output_text += "\n\n"
                f.write(output_text)

        p, r, f1, _ = macro_f1(preds, eval_labels)
        logging.info(
            "%s-%s precision: %s - recall: %s - macro f1 score: %s",
            self.task_name,
            self.my_model_name,
            p,
            r,
            f1,
        )
        return f1

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.output_dir, f"checkpoint-{self.my_model_name}-{step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        self.tokenizer.save_vocabulary(
            save_directory=output_dir, filename_prefix=self.my_model_name
        )
        logging.info("Saving models checkpoint to %s", output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model.load_pretrained(
            os.path.join(self.output_dir, f"checkpoint-{self.my_model_name}-{best_step}")
        )
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_vocabulary(
            save_directory=self.output_dir, filename_prefix=self.my_model_name
        )
        logging.info("Saving models checkpoint to %s", self.output_dir)

    def training_step(self, model, item):
        model.train()

        input_ids, token_type_ids, attention_mask, labels = self.unpack_data_item(item)

        outputs = model(
            labels=labels,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    def get_eval_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = EVAL_BATCH_SIZE

        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
