#!/usr/bin/env python3
"""Dataset class for pt -> en machine translation"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps the pt/en dataset for
     a Transformer"""

    def __init__(self, batch_size, max_len):
        """
        batch_size: batch size for training/validation
        max_len: max numberoftokens allowed per sentence

        Sets:
        data_train - train split, batched, ready for training
            data_valid - validation split, batched
            tokenizer_pt - Portuguese sub-word tokenizer
            tokenizer_en - English sub-word tokenizer
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(pt, en):
            return tf.logical_and(tf.size(pt) <= max_len,
                                   tf.size(en) <= max_len)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for dataset

        data: tf.data.Dataset of (pt, en)
        tf.Tensor pairs

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens, with start/end added

        pt: tf.Tensor containing the Portuguese sentence
        en: tf.Tensor containing the English sentence

        Returns: pt_tokens, en_tokens (lists of ints)
        """
        vocab_size = 2 ** 13

        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False)

        pt_tokens = [vocab_size] + pt_tokens + [vocab_size + 1]
        en_tokens = [vocab_size] + en_tokens + [vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for
         the encode instance method

        pt: tf.Tensor containing the Portuguese sentence
        en: tf.Tensor containing the English sentence

        Returns: pt_tokens, en_tokens (tf.Tensor)
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
