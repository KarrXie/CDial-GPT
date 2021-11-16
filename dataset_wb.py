# -*- coding: utf-8 -*-
# Some functions come from the Internet, if you violate your rights, please contact us.
import os
from itertools import chain
import json
from typing import Optional
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizer

IGNORED_ID = -1
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "speaker1", "speaker2"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def deal_fake_unk(text, token_list):
    if token_list[0] == '[UNK]' and text[:7] == 'speaker':
        if text[:8] == 'speaker1':
            token_list.insert(0, 'speaker1')
        else:
            token_list.insert(0, 'speaker2')
    return token_list


class WBDataset(Dataset):

    def __init__(self, data, tokenizer, batch_first=True, lm_labels=True, training=True):
        self.data = data
        self.tokenizer = tokenizer
        self.training = training
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.bos, self.eos, self.patient, self.doctor = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.dialog_tokenize(self.data[index])
        if self.lm_labels:
            history = data[:-1]
            resposne = data[-1]
        else:
            history = data
            resposne = []
        return self.process(history, resposne)

    def process(self, history, response, with_eos=True):
        if self.training:
            sequence = [[self.bos]] + history + [response + ([self.eos] if with_eos else [])]
        elif self.lm_labels:
            sequence = [[self.bos]] + history + [[self.doctor]]
            response = self.tokenizer.decode(response[1:])  # 剔除第一个用户标识符的token
        else:
            sequence = [[self.bos]] + history + [[self.doctor]]
            response = None
        instance = {"input_ids": list(chain(*sequence)),
                    "token_type_ids": [self.bos] + [s[0] for s in sequence[1:] for _ in s]}
        if self.training:
            instance["lm_labels"] = ([IGNORED_ID] * sum(len(s) for s in sequence[:-1])) + [IGNORED_ID] + sequence[-1][
                                                                                                         1:]
        else:
            instance["lm_labels"] = response

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        input_mask = input_ids != self.pad
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        if self.training:
            labels = pad_sequence(
                [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=IGNORED_ID)
        else:
            labels = [instance["lm_labels"] for instance in batch]
        return input_ids, token_type_ids, input_mask, labels

    def dialog_tokenize(self, obj):
        if isinstance(obj, str):
            obj_token = self.tokenizer.tokenize(obj)
            return self.tokenizer.convert_tokens_to_ids(deal_fake_unk(obj, obj_token))
        elif isinstance(obj, list):
            return list(self.dialog_tokenize(o) for o in obj)
        else:
            raise ValueError("obj should be type str or list")

class DialogDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.num_workers = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.dataset_path = args.dataset_path
        self.tokenizer = BertTokenizer.from_pretrained(
            args.model_name_or_path, do_lower_case=True, never_split=["speaker1", "speaker2"])
        self.distributed = True if args.gpus > 1 else False
        self.replace_sampler = args.replace_sampler_ddp
        self.train_data_size = None

    def setup(self, stage: Optional[str] = None) -> None:
        with open(self.dataset_path, "r", encoding="utf8") as f:
            datasets = json.loads(f.read())
        self.train_data_size = len(datasets["train"])
        self.train_dataset = WBDataset(datasets["train"], self.tokenizer)
        self.valid_dataset = WBDataset(datasets["valid"], self.tokenizer, training=False)
        self.test_dataset = WBDataset(datasets["test"], self.tokenizer, training=False)
        # self.train_sampler = torch.utils.data.distributed.DistributedSampler()
        # TODO 是否需要加上train_sampler

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   collate_fn=self.train_dataset.collate,
                                                   num_workers=self.num_workers,
                                                   batch_size=self.train_batch_size,
                                                   pin_memory=True)
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   collate_fn=self.valid_dataset.collate,
                                                   num_workers=self.num_workers,
                                                   batch_size=self.valid_batch_size,
                                                   pin_memory=True)
        return valid_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  collate_fn=self.test_dataset.collate,
                                                  num_workers=self.num_workers,
                                                  batch_size=self.valid_batch_size,
                                                  pin_memory=True)
        return test_loader
