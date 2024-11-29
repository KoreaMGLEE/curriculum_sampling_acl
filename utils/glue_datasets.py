import os
import json
import random
import requests
import jsonlines
import logging
import numpy as np

from typing import List
from os.path import dirname, exists, join
from collections import namedtuple
from torch.utils.data import Dataset

Dataset_Path = '/mnt/user3/dataset/'
HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

LABELS = {
    "NLI": ["contradiction", "entailment", "neutral"],
    "QQP": ['is_duplicate', 'is_not_duplicate'],
    "QNLI": ["entailment", "not_entailment"],
    "SST": ["negative", "positive"],
    "FEVER": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
    "RTE": ["entailment", "not_entailment"],
}

LABEL_MAPS = {key: {k: i for i, k in enumerate(labels)} for key, labels in LABELS.items()}
LABEL_MAPS["NLI"]["hidden"] = LABEL_MAPS["NLI"]["entailment"]
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(LABELS["NLI"])}


def ensure_mnli_is_downloaded():
    mnli_source = os.path.join(Dataset_Path, "glue_multinli")
    if exists(mnli_source) and len(os.listdir(mnli_source)) > 0:
        return
    else:
        raise Exception("Download MNLI from Glue and put files under glue_multinli")

def ensure_dir_exists(filename):
    """Make sure the parent directory of `filename` exists"""
    os.makedirs(dirname(filename), exist_ok=True)

def download_to_file(url, output_file):
    """Download `url` to `output_file`, intended for small files."""
    ensure_dir_exists(output_file)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(r.content)

def load_mnli(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:
    ensure_mnli_is_downloaded()
    if is_train:
        filepath = os.path.join(Dataset_Path, "glue_multinli",  "train.tsv") 
    else:
        filepath = os.path.join(Dataset_Path, "glue_multinli",  "dev_matched.tsv") 


    logging.info("Loading mnli " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)
    out = []
    for line in lines:
        line = line.split("\t")
        out.append(
            TextPairExample(line[0], line[8], line[9], LABEL_MAPS["NLI"][line[-1].rstrip()]))

    return out

def load_sst(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filepath = os.path.join(Dataset_Path, "SST-2",  "train.tsv") 
    else:
        if custom_path is None:
            filepath = os.path.join(Dataset_Path, "SST-2",  "dev.tsv") 
        else:
            filepath = join(custom_path)

    logging.info("Loading SST-2 " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)

    out = []
    for idx, line in enumerate(lines):
        line = line.split("\t")
        out.append(
            TextPairExample(idx, line[0], None, int(line[-1].rstrip())))

    return out

def load_mrpc(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filepath = os.path.join(Dataset_Path, "MRPC",  "train.txt") 
    else:
        filepath = os.path.join(Dataset_Path, "MRPC",  "test.txt") 

    logging.info("Loading MRPC " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)
    out = []

    for idx, line in enumerate(lines):
        line = line.split("\t")
        out.append(
            TextPairExample(idx, line[3], line[4], int(line[0])))

    return out

def load_qnli(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filepath = os.path.join(Dataset_Path, "QNLI",  "train.tsv")    
    else:
        if custom_path is None:
            filepath = os.path.join(Dataset_Path, "QNLI",  "dev.tsv")    


    logging.info("Loading QNLI " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)
    out = []
    for idx, line in enumerate(lines):
        line = line.split("\t")
        out.append(
            TextPairExample(idx, line[1], line[2], LABEL_MAPS["QNLI"][line[-1].rstrip()]))
    return out

def load_rte(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:    
    if is_train:
        filepath = os.path.join(Dataset_Path, "RTE",  "train.tsv")    
    else:
        filepath = os.path.join(Dataset_Path, "RTE",  "dev.tsv")    


    logging.info("Loading rte " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)
    out = []
    for idx, line in enumerate(lines):
        line = line.split("\t")
        out.append(
            TextPairExample(idx, line[1], line[2], LABEL_MAPS["RTE"][line[-1].rstrip()]))
    return out

def load_cola(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:    
    if is_train:
        filepath = os.path.join(Dataset_Path, "CoLA",  "train.tsv")           
    else:
        filepath = os.path.join(Dataset_Path, "CoLA",  "dev.tsv")   

    logging.info("Loading cola " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)
    out = []
    for idx, line in enumerate(lines):
        line = line.split("\t")

        out.append(
            TextPairExample(idx, line[3], None, line[1]))
    return out

def load_sts(is_train, seed=None, sample=None, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filepath = os.path.join(Dataset_Path, "STS-B",  "train.tsv")        
    else:
        if custom_path is None:
            filepath = os.path.join(Dataset_Path, "STS-B",  "dev.tsv")


    logging.info("Loading sts " + ("train" if is_train else "dev"))
    with open(filepath) as f:
        f.readline()
        lines = f.readlines()
    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)
    out = []
    for idx, line in enumerate(lines):
        line = line.split("\t")

        out.append(
            TextPairExample(idx, line[3], None, line[1]))
    return out

def load_hans(n_samples=None, filter_label=None, filter_subset=None) -> List[
    TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    filepath = os.path.join(Dataset_Path, 'hans', "heuristics_evaluation_set.txt")
    if not exists(filepath):
        logging.info("Downloading source to %s..." % filepath)
        download_to_file(HANS_URL, filepath)

    with open(filepath, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = 0
        elif label == "entailment":
            label = 1

        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out

def load_qqp_paws(is_train, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filepath = os.path.join(Dataset_Path, 'PAWS', 'train.tsv')
    else:
        if custom_path is None:
            filepath = os.path.join(Dataset_Path, 'PAWS', 'dev_and_test.tsv')
            
        else:
            filename = custom_path

    with open(filepath) as f:
        f.readline()
        lines = f.readlines()

    out = []
    for line in lines:
        line = line.split("\t")
        out.append(TextPairExample(line[0], eval(line[1]).decode('utf-8'), eval(line[2]).decode('utf-8'), int(line[3])))
    return out

def load_fever(is_train, seed=111, custom_path=None, sample=None):
    out = []
    if custom_path is not None:
        full_path = custom_path
    elif is_train:
        full_path = os.path.join(Dataset_Path, 'FEVER', 'nli.train.jsonl')
    else:
        full_path = os.path.join(Dataset_Path, 'FEVER', 'nli.dev.jsonl')
    logging.info("Loading jsonl from {}...".format(full_path))
    with open(full_path, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)
            id = str(i)
            text_a = example["claim"]
            text_b = example["evidence"] if "evidence" in example.keys() else example["evidence_sentence"]
            label = example["gold_label"] if "gold_label" in example.keys() else example["label"]
            out.append(TextPairExample(id, text_a, text_b, LABEL_MAPS["FEVER"][label]))
    if sample:
        random.seed(seed)
        random.shuffle(out)
        out = out[:sample]
    return out



def load_qqp(is_train, seed=111, sample=None, custom_path=None) -> List[TextPairExample]:
    out = []
    if is_train:
        filename = os.path.join(Dataset_Path, "QQP", "qqp.train.jsonl")
    else:
        if custom_path is None:
            filename = os.path.join(Dataset_Path, "QQP", "qqp.val.jsonl")
        else:
            filename = os.path.join(Dataset_Path, "QQP", custom_path)

    with jsonlines.open(filename) as f:
        for row in f.iter():
            out.append(TextPairExample(row['id'], row['sentence1'], row['sentence2'], int(row['is_duplicate'])))
    if sample:
        random.seed(seed)
        random.shuffle(out)
        out = out[:sample]
    return out