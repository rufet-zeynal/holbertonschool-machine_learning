#!/usr/bin/env python3
"""Loads and preps the pt-to-en translation dataset."""
from setup import load_pt2en
import transformers


class Dataset:
    """Loads and preps a dataset for machine translation."""

    def __init__(self):
        """Loads train/valid splits and builds the tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset.

        data is a tf.data.Dataset of (pt, en) tf.string pairs.
        Returns tokenizer_pt, tokenizer_en.
        """
        # pull every sentence out of the tf.data.Dataset first
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # start from the pretrained tokenizers, then retrain on our corpus
        pt_base = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        en_base = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        tokenizer_pt = pt_base.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13)
        tokenizer_en = en_base.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en
