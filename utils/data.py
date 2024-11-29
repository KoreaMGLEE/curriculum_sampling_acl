import re 
import os
import torch
import string
import numpy as np
from typing import List, Iterable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import namedtuple

from .processors import Processor, process_par
from .glue_datasets import * 
from .collator import collate_input_features, GPT_collate_input_features, GPT_collate_eval_input_features, T5_collate_input_features, llama_collate_input_features, llama_collate_eval_input_features


TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

task_loaders = {
    'MNLI': load_mnli(True),
    'FEVER': load_fever(True),
    'QQP': load_qqp(True),
    'SST': load_sst(True),
    'MRPC': load_mrpc(True),
    'QNLI': load_qnli(True),
    'RTE': load_rte(True),
    'COLA': load_cola(True),
    'STS': load_sts(True)
}

eval_loaders = {
    'MNLI': [("mnli_dev_m", load_mnli(False)), ("hans", load_hans())],
    'FEVER': [("fever_dev", load_fever(False)),],
            #   ("fever_symmetric_dev", load_fever(False, custom_path= os.path.join(Dataset_Path, 'FEVER-symmetric-generated', 'fever_symmetric_dev.jsonl'))),
            #   ("fever_symmetric_test", load_fever(False, custom_path= os.path.join(Dataset_Path, 'FEVER-symmetric-generated', 'fever_symmetric_test.jsonl')))],
    'QQP': [("qqp_dev", load_qqp(False)), ("qqp_paws", load_qqp_paws(False))],
    'SST': [("sst_dev", load_sst(False))],
    'MRPC': [("mrpc_dev", load_mrpc(False))],
    'QNLI': [("qnli_dev", load_qnli(False))],
    'RTE': [("rte_dev", load_rte(False))],
    'COLA': [("cola_dev", load_cola(False))],
    'STS': [("sts_dev", load_sts(False)), ("sts_train", load_sts(True))]
}

label_map = {
    'MNLI': {1: 'entailment', 0: 'contradiction', 2: 'neutral'},
    'QQP': {0: 'is_duplicate', 1: 'is_not_duplicate'},
    'FEVER': {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def score(metric, preds, answers, is_wsc=False, model='t5-base'):
    if metric == 'acc':
        if 'gpt' in model or 'llama' in model.lower():
            correct = sum(a in p.split('answer')[-1] for p, a in zip(preds, answers))
        else:
            correct = sum(a == b for a, b in zip(preds, answers))

        total_set = {}
        false_set = {}
        for i, p in enumerate(answers):
            total_set[p] = total_set.get(p, 0) + 1
            if (('gpt' in model or 'llama' in model.lower()) and p not in preds[i]) or (p != preds[i]):
                false_set[p] = false_set.get(p, 0) + 1

        result = correct / len(answers)
    return result

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, example_id, input_ids, segment_ids, label_id):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id

def template_function(task, premise, hypothesis):
    if task == 'MNLI':
        return f"mnli hypothesis: {hypothesis} premise: {premise} answer: "
    elif task == 'QQP':
        return f"qqp question1: {premise} question2: {hypothesis} answer:  "
    elif task == 'FEVER':
        return f"fever claim: {hypothesis} evidence: {premise} answer: "
    elif task == 'SST':
        return f"sst sentence: {premise} answer: "
    elif task == 'MRPC':
        return f"mrpc sentence1: {premise} sentence2: {hypothesis} answer: "
    elif task == 'QNLI':
        return f"qnli question: {hypothesis} sentence: {premise} answer: "
    elif task == 'RTE':
        return f"rte hypothesis: {hypothesis} premise: {premise} answer: "
    elif task == 'COLA':
        return f"cola sentence: {premise} answer: "
    elif task == 'STS':
        return f"sts sentence1: {premise} sentence2: {hypothesis} answer: "
    else:
        return f"{premise} {hypothesis}"



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []

        for example in data:
            tokens_a, tokens_b = self.tokenizer.tokenize(example.premise), None
            if example.hypothesis:
                tokens_b = self.tokenizer.tokenize(example.hypothesis)
                _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.max_seq_length - 2)]

            tokens = [self.tokenizer.cls_token] + tokens_a
            if tokens_b:
                tokens += tokens_b + [self.tokenizer.sep_token]
            
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                ))

        return features


class Seq2Seq_ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.task = 'MNLI'

def process(self, data: Iterable):
    features = []

    for example in data:
        tokens = self.tokenizer.tokenize(example.premise)
        if example.hypothesis:
            tokens += self.tokenizer.tokenize(example.hypothesis)
        tokens = tokens[:self.max_seq_length - 1] + [self.tokenizer.eos_token]
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label = label_map[self.task].get(int(example.label), 'neutral')
        label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{label}</s>"))

        features.append(
        InputFeatures(
            example_id=example.id,
            input_ids=np.array(input_ids),
            segment_ids=np.array(segment_ids),
            label_id=label,
        ))

    return features

class Decoder_ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer, data_type='train'):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.task = 'MNLI'
        self.data_type = data_type

    def process(self, data: Iterable):
        features = []
        for idx, example in enumerate(data):
            prompt_sentence = template_function(self.task, example.premise, example.hypothesis)
            tokens = self.tokenizer.encode(prompt_sentence)
            segment_ids = [0] * len(tokens)

            label = label_map.get(self.task, {}).get(int(example.label), example.label)
            if self.tokenizer.eos_token_id == 128001:
                label = self.tokenizer.encode(f"{label}")[1:] + [self.tokenizer.eos_token_id]
            else:
                label = self.tokenizer.encode(f"{label}") + [self.tokenizer.eos_token_id]

            if self.data_type == 'train':                
                tokens = tokens + label 
                segment_ids += [0] * (len(label))

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(tokens),
                    segment_ids=np.array(segment_ids),
                    label_id=label,
                ))

        return features



def convert_examples_to_features(
    examples: List[TextPairExample], max_seq_length, tokenizer, n_process=4, task='MNLI', model_name_or_path='t5', data_type='train'):
    if 't5' in model_name_or_path:
        converter = Seq2Seq_ExampleConverter(max_seq_length, tokenizer)
    elif 'gpt' in model_name_or_path or 'llama' in model_name_or_path.lower():
        converter = Decoder_ExampleConverter(max_seq_length, tokenizer, data_type=data_type)
    else:
        converter = ExampleConverter(max_seq_length, tokenizer)
    
    converter.task = task
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")


def load_data(task_name, split, model_name_or_path, tokenizer, n_processes=4, max_seq_length=128):
    if split == 'train':
        train_examples = task_loaders[task_name]

        train_features: List[InputFeatures] = convert_examples_to_features(train_examples[:50000], max_seq_length,
                                                                               tokenizer, n_processes, model_name_or_path=model_name_or_path)
        return train_features

    elif split == 'eval':    
        eval_examples, eval_features, all_label_ids, dataset_names = [], [], [], []

        if task_name in eval_loaders:
            eval_examples += eval_loaders[task_name]

        for name, eval_example in eval_examples:
            eval_feature = convert_examples_to_features(
                eval_example[:5000], max_seq_length, tokenizer, n_process=n_processes, task=task_name, model_name_or_path=model_name_or_path, data_type='eval')
        
            eval_feature.sort(key=lambda x: len(x.input_ids))
            all_label_ids.append(np.array([item for sublist in [x.label_id for x in eval_feature] for item in sublist]))
            dataset_names.append(name)
            eval_features.append(eval_feature)

        return eval_features, all_label_ids, dataset_names
    

class InputFeatureDataset(Dataset):

    def __init__(self, examples: List[InputFeatures]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def build_dataloader(data, batch_size, seed=None, model_name_or_path=None, is_train=True):
    ds = InputFeatureDataset(data)
    sampler = RandomSampler(ds) if is_train else SequentialSampler(ds)

    if 't5' in model_name_or_path:
        collate_fn = T5_collate_input_features
    elif 'gpt' in model_name_or_path:
        collate_fn = GPT_collate_input_features if is_train else GPT_collate_eval_input_features
    elif 'llama' in model_name_or_path.lower():
        collate_fn = llama_collate_input_features if is_train else llama_collate_eval_input_features
    else:
        collate_fn = collate_input_features

    return DataLoader(ds, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

