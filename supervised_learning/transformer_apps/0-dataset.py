#!/usr/bin/env python3
"""
Sets up the dataset and tokenizers for machine translation
"""
import tensorflow as tf
from transformers import AutoTokenizer
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
        pt_base_tokenizer = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        en_base_tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )

        def get_pt_corpus():
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in pt]

        def get_en_corpus():
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in en]

        vocab_size = 2 ** 13
        tokenizer_pt = pt_base_tokenizer.train_new_from_iterator(
            get_pt_corpus(),
            vocab_size=vocab_size
        )
        tokenizer_en = en_base_tokenizer.train_new_from_iterator(
            get_en_corpus(),
            vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
