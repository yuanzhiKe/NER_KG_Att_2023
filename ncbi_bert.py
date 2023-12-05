import os
import sys
import logging
import datasets
from transformers import AutoTokenizer
from models.base_model import NER_Model
from models.ner_models import NER_Model_SA, NER_Model_OA
from configs.ncbi_config import model_name, OUTPUT_DIR, MAX_LEN, DEVICE
from train.trainer import NCBITrainer
from data_interfaces.ncbidata import NCBIDataset
from tree.english_ontology import EN_Ontology_tree
from datetime import datetime
from logging import StreamHandler, FileHandler, Formatter
from logging import INFO, DEBUG, NOTSET


def perform_base():
    my_model_name = "NCBI_BERT_BASE"
    dataset = datasets.load_dataset("ncbi_disease")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NCBIDataset(dataset["train"], tokenizer, MAX_LEN)
    eval_dataset = NCBIDataset(dataset["validation"], tokenizer, MAX_LEN)
    test_dataset = NCBIDataset(dataset["test"], tokenizer, MAX_LEN)

    model = NER_Model(model_name, num_labels=3, my_model_name=my_model_name)
    trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
    )
    _, best_step = trainer.train()
    model = model.load_pretrained(
        os.path.join(OUTPUT_DIR, f"checkpoint-{my_model_name}-{best_step}")
    )
    test_trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
    )
    test_trainer.evaluate(model=model)
    logging.info("Base model tested.")


def perform_sa():
    my_model_name = "NCBI_BERT_SA"
    dataset = datasets.load_dataset("ncbi_disease")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NCBIDataset(dataset["train"], tokenizer, MAX_LEN)
    eval_dataset = NCBIDataset(dataset["validation"], tokenizer, MAX_LEN)
    test_dataset = NCBIDataset(dataset["test"], tokenizer, MAX_LEN)

    model = NER_Model_SA(model_name, num_labels=3, my_model_name=my_model_name)
    trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
    )
    _, best_step = trainer.train()
    model = model.load_pretrained(
        os.path.join(OUTPUT_DIR, f"checkpoint-{my_model_name}-{best_step}")
    )
    test_trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
    )
    test_trainer.evaluate(model=model)
    logging.info("SA model tested.")


def perform_oa():
    ontology = EN_Ontology_tree().get_dfs_encoded_str()
    my_model_name = "NCBI_BERT_OA"
    dataset = datasets.load_dataset("ncbi_disease")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NCBIDataset(dataset["train"], tokenizer, MAX_LEN)
    eval_dataset = NCBIDataset(dataset["validation"], tokenizer, MAX_LEN)
    test_dataset = NCBIDataset(dataset["test"], tokenizer, MAX_LEN)

    model = NER_Model_OA(
        model_name,
        num_labels=3,
        my_model_name=my_model_name,
        ontology=ontology,
        tokenizer=tokenizer,
        device=DEVICE,
    )

    trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
    )
    _, best_step = trainer.train()
    model = model.load_pretrained(
        os.path.join(OUTPUT_DIR, f"checkpoint-{my_model_name}-{best_step}")
    )
    test_trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
    )
    test_trainer.evaluate(model=model)
    logging.info("OA model tested.")


if __name__ == "__main__":
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter("%(message)s"))

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_handler = FileHandler(
        os.path.join(OUTPUT_DIR, f"log{datetime.now():%Y%m%d%H%M%S}.log")
    )
    file_handler.setLevel(INFO)
    file_handler.setFormatter(
        Formatter("%(asctime)s@ %(name)s [%(levelname)s] %(funcName)s: %(message)s")
    )

    logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])
    perform_base()
    perform_sa()
    perform_oa()
