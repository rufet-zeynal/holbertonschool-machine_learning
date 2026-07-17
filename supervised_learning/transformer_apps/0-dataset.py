#!/usr/bin/env python3
"""Module for loading and preparing a machine translation dataset."""
import transformers
from setup import load_pt2en
class Dataset:
    """Loads and preps a dataset for machine translation."""
    def __init__(self):
        """Initialize the Dataset instance.
        Creates instance attributes:
            data_train: the training split of ted_hrlr_translate/pt_to_en
            data_valid: the validation split of ted_hrlr_translate/pt_to_en
            tokenizer_pt: the Portuguese sub-word tokenizer
            tokenizer_en: the English sub-word tokenizer
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
    def tokenize_dataset(self, data):
        """Create sub-word tokenizers for the dataset.
        Args:
            data: a tf.data.Dataset whose examples are (pt, en) tuples
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing the English sentence
        Returns:
            tokenizer_pt: the Portuguese tokenizer
            tokenizer_en: the English tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        def pt_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, en in data:
                yield pt.numpy().decode('utf-8')
        def en_iterator():
            """Yield English sentences from the dataset."""
            for pt, en in data:
                yield en.numpy().decode('utf-8')
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2 ** 13
        )
        return tokenizer_pt, tokenizer_en
