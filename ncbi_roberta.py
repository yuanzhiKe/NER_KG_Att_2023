import os
import subprocess


def download_cblue_package():
    project_path = os.path.dirname(__file__)
    cblue_path = os.path.join(project_path, "CBLUE")
    if not os.path.exists(cblue_path):
        subprocess.call(
            ["git", "clone", "git@github.com:CBLUEbenchmark/CBLUE.git", cblue_path]
        )
    return cblue_path


cblue_path = download_cblue_package()

import sys

sys.path.append(os.path.join(cblue_path))

import logging
import datasets
from transformers import AutoTokenizer
from models.base_model import NER_Model
from models.ner_models import NER_Model_SA, NER_Model_OA
from configs.ncbi_roberta_config import model_name, OUTPUT_DIR, MAX_LEN, DEVICE
from train.trainer import NCBITrainer
from data_interfaces.ncbidata import NCBIDataset
from tree.english_ontology import EN_Ontology_tree
from datetime import datetime
from logging import StreamHandler, FileHandler, Formatter
from logging import INFO, NOTSET

logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)


def read_dataset(dataset_name, tokenizer, is_roberta):
    dataset = datasets.load_dataset(dataset_name)
    train_dataset = NCBIDataset(
        dataset["train"], tokenizer, MAX_LEN, is_roberta=is_roberta
    )
    eval_dataset = NCBIDataset(
        dataset["validation"], tokenizer, MAX_LEN, is_roberta=is_roberta
    )
    test_dataset = NCBIDataset(
        dataset["test"], tokenizer, MAX_LEN, is_roberta=is_roberta
    )
    return train_dataset, eval_dataset, test_dataset


def perform_base():
    my_model_name = "NCBI_ROBERTA_BASE"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset, test_dataset = read_dataset(
        "ncbi_disease", tokenizer, True
    )

    model = NER_Model(model_name, num_labels=3, my_model_name=my_model_name)
    trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
        is_roberta=True,
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
        is_roberta=True,
    )
    test_trainer.evaluate(model=model)
    logging.info("Base model tested.")


def perform_sa():
    my_model_name = "NCBI_ROBERTA_SA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset, test_dataset = read_dataset(
        "ncbi_disease", tokenizer, True
    )

    model = NER_Model_SA(model_name, num_labels=3, my_model_name=my_model_name)
    trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
        is_roberta=True,
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
        is_roberta=True,
    )
    test_trainer.evaluate(model=model)
    logging.info("SA model tested.")


def perform_oa(ontology_order="Pre"):
    if ontology_order == "Pre":
        ontology = EN_Ontology_tree().get_dfs_encoded_str()
    elif ontology_order == "Post":
        ontology = EN_Ontology_tree().get_post_order_encoded_str()
    elif ontology_order == "Bi":
        ontology = EN_Ontology_tree().get_pre_post_order_encoded_str()
    else:
        logging.warn("Unavailabe ontology order, using pre-order DFS")
        ontology = EN_Ontology_tree().get_dfs_encoded_str()

    my_model_name = "NCBI_ROBERTA_OA" + " " + ontology_order
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset, test_dataset = read_dataset(
        "ncbi_disease", tokenizer, True
    )

    model = NER_Model_OA(
        model_name,
        num_labels=3,
        my_model_name=my_model_name,
        ontology=ontology,
        tokenizer=tokenizer,
        device=DEVICE,
        is_roberta=True,
    )

    trainer = NCBITrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="ncbi",
        my_model_name=my_model_name,
        is_roberta=True,
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
        is_roberta=True,
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
    # perform_base()
    # perform_sa()
    # perform_oa()
    perform_oa("Post")
    perform_oa("Bi")
