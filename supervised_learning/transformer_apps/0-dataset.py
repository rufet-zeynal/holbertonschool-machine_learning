#!/usr/bin/env python3
"""
Sets up the dataset and tokenizers for machine translation
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Dataset class that loads and prepares a dataset for machine translation
    """
    def __init__(self):
        """
        Initializes the dataset object and sets up the tokenizers.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        pt_tok, en_tok = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt_tok
        self.tokenizer_en = en_tok

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset
        """
        pt_base = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        en_base = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )

        # Safely extract text without relying on as_numpy_iterator()
        pt_list = []
        en_list = []
        for pt, en in data:
            pt_list.append(pt.numpy().decode('utf-8'))
            en_list.append(en.numpy().decode('utf-8'))

        vocab_size = 2 ** 13
        tokenizer_pt = pt_base.train_new_from_iterator(
            pt_list,
            vocab_size=vocab_size
        )
        tokenizer_en = en_base.train_new_from_iterator(
            en_list,
            vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
