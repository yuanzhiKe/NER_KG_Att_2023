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
import argparse
from transformers import AutoTokenizer
from models.base_model import NER_Model
from models.ner_models import NER_Model_SA, NER_Model_OA
from configs.CMeEE_config import model_name, OUTPUT_DIR, DEVICE, TASK_DATA_DIR
from train.cblue_trainer import CBLUETrainer
from data_interfaces.CMeEEdata import MyEEDataset
from tree.ontology import CBLUE_Ontology
from datetime import datetime
from logging import StreamHandler, FileHandler, Formatter
from logging import INFO, NOTSET


def get_cblue_datasets(tokenizer):
    train_dataset = MyEEDataset(data_path=TASK_DATA_DIR, 
                                tokenizer=tokenizer,
                                type="train",
                                mode="train"
                                )
    eval_dataset = MyEEDataset(data_path=TASK_DATA_DIR, 
                                tokenizer=tokenizer,
                                type="eval",
                                mode="train"
                                )
    test_dataset = MyEEDataset(data_path=TASK_DATA_DIR, 
                                tokenizer=tokenizer,
                                type="eval",
                                mode="train"
                                )
    num_labels = train_dataset.get_data_processor().num_labels
    return train_dataset, eval_dataset, test_dataset, num_labels

def get_trainer(model, tokenizer, train_dataset, eval_dataset, my_model_name, metric):
    trainer = CBLUETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_name="cblue",
        my_model_name=my_model_name,
        metric=metric
    )
    return trainer

def perform_base(args):
    my_model_name = "CBLUE_BERT_BASE"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset, test_dataset, num_labels = get_cblue_datasets(tokenizer)

    model = NER_Model(model_name, num_labels=num_labels, my_model_name=my_model_name)
    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset, my_model_name, args.metric)
    _, best_step = trainer.train()
    model = model.load_pretrained(
        os.path.join(OUTPUT_DIR, f"checkpoint-{my_model_name}-{best_step}")
    )
    test_trainer = CBLUETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        task_name="cblue",
        my_model_name=my_model_name,
    )
    test_trainer.evaluate(model=model)
    logging.info("Base model tested.")


def perform_sa(args):
    my_model_name = "CBLUE_BERT_SA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset, test_dataset, num_labels = get_cblue_datasets(tokenizer)

    model = NER_Model_SA(model_name, num_labels=num_labels, my_model_name=my_model_name)
    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset, my_model_name, args.metric)
    _, best_step = trainer.train()
    model = model.load_pretrained(
        os.path.join(OUTPUT_DIR, f"checkpoint-{my_model_name}-{best_step}")
    )
    test_trainer = CBLUETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        task_name="cblue",
        my_model_name=my_model_name,
    )
    test_trainer.evaluate(model=model)
    logging.info("SA model tested.")


def perform_oa(args, ontology_order="Pre"):
    if ontology_order == "Pre":
        ontology = CBLUE_Ontology().get_dfs_encoded_str()
    elif ontology_order == "Post":
        ontology = CBLUE_Ontology().get_post_order_encoded_str()
    elif ontology_order == "Bi":
        ontology = CBLUE_Ontology().get_pre_post_order_encoded_str()
    else:
        logging.warn("Unavailabe ontology order, using pre-order DFS")
        ontology = CBLUE_Ontology().get_dfs_encoded_str()

    my_model_name = "CBLUE_BERT_OA" + "_" + ontology_order
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset, test_dataset, num_labels = get_cblue_datasets(tokenizer)

    model = NER_Model_OA(
        model_name,
        num_labels=num_labels,
        my_model_name=my_model_name,
        ontology=ontology,
        tokenizer=tokenizer,
        device=DEVICE,
    )

    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset, my_model_name, args.metric)
    _, best_step = trainer.train()
    model = model.load_pretrained(
        os.path.join(OUTPUT_DIR, f"checkpoint-{my_model_name}-{best_step}")
    )
    test_trainer = CBLUETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        task_name="cblue",
        my_model_name=my_model_name,
    )
    test_trainer.evaluate(model=model)
    logging.info("OA model tested.")


def get_outputs(model, tokenizer, test_dataset):
    test_trainer = CBLUETrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset=test_dataset,
        task_name="cblue",
        my_model_name=model.my_model_name,
    )
    test_trainer.evaluate(model=model, output_results=True)


def output_pick_comp():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _, _, test_dataset, num_labels = get_cblue_datasets(tokenizer)

    base_model = NER_Model(model_name, num_labels=num_labels, my_model_name="CBLUE_BERT_BASE")
    sa_model = NER_Model_SA(model_name, num_labels=num_labels, my_model_name="CBLUE_BERT_SA")
    oa_model = NER_Model_OA(
        model_name,
        num_labels=3,
        my_model_name="CBLUE_BERT_OA_Pre",
        ontology=CBLUE_Ontology().get_dfs_encoded_str(),
        tokenizer=tokenizer,
        device=DEVICE,
    )
    base_model.load_pretrained(OUTPUT_DIR)
    sa_model.load_pretrained(OUTPUT_DIR)
    oa_model.load_pretrained(OUTPUT_DIR)

    get_outputs(base_model, tokenizer, test_dataset)
    get_outputs(sa_model, tokenizer, test_dataset)
    get_outputs(oa_model, tokenizer, test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', help='availabe option for metric: "macro" for macro F1, "micro" for micro F1.')
    args = parser.parse_args()

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
    perform_base(args)
    # perform_sa(args)
    perform_oa(args)
    perform_oa(args, "Post")
    perform_oa(args, "Bi")
    # output_pick_comp()
