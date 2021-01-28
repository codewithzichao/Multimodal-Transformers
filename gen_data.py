import os
import json
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer
from torchvision.transforms import transforms
from args import MyArgs
from preprocessing import zip_image_text_label


class MyDataset(Dataset):
    def __init__(self, data, label2idx, tokenizer, max_length=256):
        super(MyDataset, self).__init__()

        self.data = data
        self.label2idx = label2idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        image = self.data[item][0]
        text = self.data[item][1]
        label = self.data[item][2]

        tokenizer_result = self.tokenizer.encode_plus(text, add_special_tokens=True, \
                                                      max_length=self.max_length, \
                                                      padding="max_length", \
                                                      return_tensors="pt")
        input_ids = tokenizer_result["input_ids"].squeeze(dim=0)
        attention_mask = tokenizer_result["attention_mask"].squeeze(dim=0)

        if input_ids.shape[-1] != self.max_length:
            input_ids = input_ids[:self.max_length]
        if attention_mask.shape[-1] != self.max_length:
            attention_mask = attention_mask[:self.max_length]

        label = self.label2idx[label]

        return image, input_ids, attention_mask, label


class MyTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        super(MyTestDataset, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        image = self.data[item][0]
        text = self.data[item][1]
        image_name = self.data[item][2]

        tokenizer_result = self.tokenizer.encode_plus(text, add_special_tokens=True, \
                                                      max_length=self.max_length, \
                                                      padding="max_length", \
                                                      return_tensors="pt")
        input_ids = tokenizer_result["input_ids"].squeeze(dim=0)
        attention_mask = tokenizer_result["attention_mask"].squeeze(dim=0)

        if input_ids.shape[-1] != self.max_length:
            input_ids = input_ids[:self.max_length]
        if attention_mask.shape[-1] != self.max_length:
            attention_mask = attention_mask[:self.max_length]

        return image, input_ids, attention_mask, image_name
