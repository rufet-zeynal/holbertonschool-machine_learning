#!/usr/bin/env python3
"""Dataset class for pt -> en machine translation"""
import transformers
from setup import load_pt2en


class Dataset:
    """Load and preps the pt/en dataset for a Transformer"""

    def __init__(self):
        """
        Sets:
            data_train - train split, (pt, en) tf.string pairs
            data_valid - validation split, (pt, en) tf.string pairs
            tokenizer_pt - Portuguese sub-word tokenizer
            tokenizer_en - English sub-word tokenizer
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset

        data: tf.data.Dataset of (pt, en) tf.Tensor pairs

        Returns: tokenizer_pt, tokenizer_en
        """
        vocab_size = 2 ** 13

        # re-fit pretrained tokenizers on our corpus
        pt_pretrained = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        en_pretrained = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        def pt_sentences():
            for pt, en in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_sentences():
            for pt, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = pt_pretrained.train_new_from_iterator(
            pt_sentences(), vocab_size=vocab_size)
        tokenizer_en = en_pretrained.train_new_from_iterator(
            en_sentences(), vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en
