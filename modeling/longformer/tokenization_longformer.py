"""Tokenization classes for Hi-Transformer."""
import torch
from transformers import AutoTokenizer
from transformers.models.longformer.configuration_longformer import LongformerConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)


class LongformerTokenizer:
    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.config = LongformerConfig.from_pretrained(self._tokenizer.name_or_path)
        # hardcoded values
        self.config.max_sentence_size = 128
        self.config.max_sentence_length = 128
        self.config.max_sentences = 32
        self._tokenizer.model_max_length = self.model_max_length
        self.type2id = {'input_ids': (self._tokenizer.sep_token_id, self._tokenizer.pad_token_id),
                        'token_type_ids': (0, 0),
                        'attention_mask': (1, 0),
                        'special_tokens_mask': (1, -100)}

    @property
    def model_max_length(self):
        return self.config.model_max_length

    @property
    def mask_token(self):
        return self._tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self._tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self._tokenizer.sep_token_id

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return len(self._tokenizer)

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self._tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def tokenize(self, text, **kwargs):
        return self._tokenizer.tokenize(text, **kwargs)

    def encode(self, text, **kwargs):
        input_ids = self._tokenizer.encode_plus(text, add_special_tokens=False, **kwargs)
        input_ids = self.chunks(input_ids[: self.model_max_length - self.config.max_sentences],
                                chunk_size=self.config.max_sentence_length, special_id=self.type2id['input_ids'])

        for idx, _ in enumerate(input_ids):
            input_ids[idx][0] = self._tokenizer.cls_token_id

        return input_ids

    def get_special_tokens_mask(self, *args, **kwargs):
        return self._tokenizer.get_special_tokens_mask(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls(tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs))

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained( *args, **kwargs)

    def __call__(self, text, **kwargs):
        if isinstance(text[0], list):
            batch = self.auto_chunking(text, **kwargs)
            for idx, _ in enumerate(batch['input_ids']):
                batch['input_ids'][idx][0] = self._tokenizer.cls_token_id
        else:
            batch = self._tokenizer(text, **kwargs)

        return batch

    def auto_chunking(self, texts, **kwargs):
        batch = {}
        for text_idx, text in enumerate(texts):
            example_batch = self._tokenizer(text, add_special_tokens=False, **kwargs)
            for input_key in example_batch:
                key_inputs_list = []
                for idx, example in enumerate(example_batch[input_key][:self.config.max_sentences]):
                    key_inputs_list.append(self.pad_sentence(example, special_id=self.type2id[input_key]))
                if isinstance(key_inputs_list[0], list):
                    key_inputs_list = [token for sentence in key_inputs_list for token in sentence]
                else:
                    key_inputs_list = torch.stack(key_inputs_list)
                if input_key in batch:
                    batch[input_key].append(key_inputs_list)
                else:
                    batch[input_key] = [key_inputs_list]

        if kwargs['padding']:
            batch = self.pad(batch,
                             padding=kwargs['padding'],
                             max_length=kwargs['max_length'],
                             pad_to_multiple_of=kwargs['max_length'])

        return batch

    def pad_sentence(self, flat_input, chunk_size=128, special_id=(0, 0)):
        if isinstance(flat_input, list):
            return [special_id[0]] + flat_input[:chunk_size-1] + [self.pad_token_id] * max(0, chunk_size - len(flat_input) - 1)
        else:
            return torch.cat((torch.tensor([special_id[0] if flat_input[:chunk_size-1].sum()
                                            else special_id[1]], dtype=torch.int),
                              flat_input[:chunk_size-1],
                              torch.tensor([self.pad_token_id] * max(0, chunk_size - len(flat_input) - 1), dtype=torch.int)
                              ))

if __name__ == "__main__":
    tokenizer = LongformerTokenizer.from_pretrained('roberta-base')
    inputs = tokenizer([' '.join(['dog'] * 8192),
                        ' '.join(['cat'] * 7000),
                       ' '.join(['mouse'] * 5000)],
                       padding=True, max_length=8192, truncation=True
                       )
    print()
