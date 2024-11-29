
import torch 
import numpy as np
from typing import List

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, example_id, input_ids, segment_ids, label_id, teacher_labels):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id

def GPT_collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = torch.full((sz, max_seq_len), 50256)
    segment_ids = torch.full((sz, max_seq_len), 0)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    label_ids = torch.full((sz, max_seq_len), -100)

    for i, ex in enumerate(batch):
        input_ids[i, max_seq_len-len(ex.input_ids):] = torch.tensor(ex.input_ids)
        segment_ids[i, max_seq_len - len(ex.segment_ids):] = torch.tensor(ex.segment_ids)
        mask[i, max_seq_len-len(ex.input_ids):] = 1
        label_ids[i, max_seq_len-len(ex.label_id):] = torch.tensor(ex.label_id)

    padded_labels = label_ids

    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = [str(x.example_id) for x in batch]

    return example_ids, input_ids, mask, segment_ids, padded_labels
    

def GPT_collate_eval_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = torch.full((sz, max_seq_len), 50256)
    segment_ids = torch.full((sz, max_seq_len), 0)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)

    label_max_length = max(len(x.label_id) for x in batch)

    for i, ex in enumerate(batch):
        input_ids[i, max_seq_len-len(ex.input_ids):] = torch.tensor(ex.input_ids)
        segment_ids[i, max_seq_len - len(ex.segment_ids):] = torch.tensor(ex.segment_ids)
        mask[i, max_seq_len-len(ex.input_ids):] = 1

    label_ids = torch.full((sz, label_max_length), -100)
    for i, ex in enumerate(batch):
        label_ids[i, label_max_length - len(ex.label_id):] = torch.tensor(ex.label_id)

    padded_labels = label_ids

    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = [str(x.example_id) for x in batch]

    return example_ids, input_ids, mask, segment_ids, padded_labels


def llama_collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = torch.full((sz, max_seq_len), 128001)
    segment_ids = torch.full((sz, max_seq_len), 0)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    label_ids = torch.full((sz, max_seq_len), -100)

    for i, ex in enumerate(batch):
        input_ids[i, max_seq_len-len(ex.input_ids):] = torch.tensor(ex.input_ids)
        segment_ids[i, max_seq_len - len(ex.segment_ids):] = torch.tensor(ex.segment_ids)
        mask[i, max_seq_len-len(ex.input_ids):] = 1
        label_ids[i, max_seq_len-len(ex.label_id):] = torch.tensor(ex.label_id)

    padded_labels = label_ids

    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = [str(x.example_id) for x in batch]

    return example_ids, input_ids, mask, segment_ids, padded_labels
    

def llama_collate_eval_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = torch.full((sz, max_seq_len), 2)
    segment_ids = torch.full((sz, max_seq_len), 0)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)

    label_max_length = max(len(x.label_id) for x in batch)

    for i, ex in enumerate(batch):
        input_ids[i, max_seq_len-len(ex.input_ids):] = torch.tensor(ex.input_ids)
        segment_ids[i, max_seq_len - len(ex.segment_ids):] = torch.tensor(ex.segment_ids)
        mask[i, max_seq_len-len(ex.input_ids):] = 1

    label_ids = torch.full((sz, label_max_length), -100)
    for i, ex in enumerate(batch):
        label_ids[i, label_max_length - len(ex.label_id):] = torch.tensor(ex.label_id)

    padded_labels = label_ids

    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = [str(x.example_id) for x in batch]

    return example_ids, input_ids, mask, segment_ids, padded_labels

def T5_collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = [x.label_id for x in batch]

    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = [str(x.example_id) for x in batch]

    label_ids = [torch.LongTensor(d) for d in label_ids]
    padded_labels = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)

    return example_ids, input_ids, mask, segment_ids, padded_labels


def collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = torch.zeros((sz, max_seq_len), dtype=torch.int64)
    segment_ids = torch.zeros((sz, max_seq_len), dtype=torch.int64)
    mask = torch.zeros((sz, max_seq_len), dtype=torch.int64)

    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = torch.tensor(ex.input_ids)
        segment_ids[i, :len(ex.segment_ids)] = torch.tensor(ex.segment_ids)
        mask[i, :len(ex.input_ids)] = 1

    label_ids = torch.tensor([x.label_id for x in batch], dtype=torch.int64)

    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch), dtype=torch.int64)

    return example_ids, input_ids, mask, segment_ids, label_ids
