import re

import torch
from transformers import AutoTokenizer


class FlotaTokenizer:
    def __init__(self, model_path, k=4, strict=False, mode='flota'):
        self.model_path = model_path
        self.k = k
        self.strict = strict
        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.vocab = self.tok.vocab
        assert len(self.vocab) == self.tok.vocab_size
        if 'bert-' in self.model_path:
            self.special = '##'
            self.max_len = 18
        elif self.model_path in ['roberta-base', 'gpt2', 'roberta-large']:
            self.special = '\u0120'
            self.max_len = 19
        elif self.model_path == 'xlm-roberta-base':
            self.special = '▁'
            self.max_len = 16

    def __call__(self, texts):
        texts = [self.encode(text) for text in texts]
        batch_size = len(texts)
        max_len = max(len(text) for text in texts)
        if 'bert-' in self.model_path:
            input_ids = torch.zeros((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, :len(text)] = torch.tensor(text)
                attention_mask[i, :len(text)] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return tensors
        elif self.model_path in ['roberta-base', 'gpt2', 'roberta-large']:
            input_ids = self.tok.eos_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return tensors
        elif self.model_path == 'xlm-roberta-base':
            input_ids = self.tok.pad_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return tensors

    def max_subword_split(self, w):
        for l in range(min(len(w), self.max_len), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == '-':
                    continue
                subword = w[i:i + l]
                if 'bert-' in self.model_path:
                    if i == 0:
                        if subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                        elif not self.strict and self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                    else:
                        if self.special + subword in self. vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                        elif subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                elif self.model_path in ['roberta-base', 'gpt2', 'roberta-large'] or self.model_path == 'xlm-roberta-base':
                    if i == 0:
                        if self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                        elif subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                    else:
                        if subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                        elif self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
        return None, None, None

    def get_flota_dict(self, w, k):
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if k == 1 or rest == len(rest) * '-':
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest, k - 1)
        flota_dict[i] = max_subword
        return flota_dict

    def tokenize(self, w):
        if 'bert-' in self.model_path:
            if w in self.vocab:
                return [w]
            elif self.special + w in self.vocab:
                return [self.special + w]
        elif self.model_path in ['roberta-base', 'gpt2', 'roberta-large'] or self.model_path == 'xlm-roberta-base':
            if self.special + w in self.vocab:
                return [self.special + w]
            elif w in self.vocab:
                return [w]
        if self.mode == 'flota':
            flota_dict = self.get_flota_dict(w, self.k)
            return [subword for i, subword in sorted(flota_dict.items())]
        elif self.mode == 'first':
            if self.model_path in ['roberta-base', 'gpt2', 'roberta-large']:
                return self.tok.tokenize(' ' + w)[:self.k]
            return self.tok.tokenize(w)[:self.k]
        elif self.mode == 'longest':
            if self.model_path in ['roberta-base', 'gpt2', 'roberta-large']:
                subwords = enumerate(self.tok.tokenize(' ' + w))
            else:
                subwords = enumerate(self.tok.tokenize(w))
            topk_subwords = sorted(subwords, key=lambda x: len(x[1].lstrip(self.special)), reverse=True)[:self.k]
            return [subword for i, subword in sorted(topk_subwords, key=lambda x: x[0])]

    def encode(self, text):
        text_split = re.findall(r'[\w]+|[^\s\w]', text)
        tokens = list()
        for w in text_split:
            tokens.extend(self.tokenize(w))
        if 'bert-' in self.model_path:
            ids_flota = self.tok.convert_tokens_to_ids(tokens)[:self.tok.model_max_length - 2]
            return [self.tok.cls_token_id] + ids_flota + [self.tok.sep_token_id]
        elif self.model_path in ['roberta-base', 'gpt2', 'roberta-large']:
            return self.tok.convert_tokens_to_ids(tokens)[:self.tok.model_max_length]
        elif self.model_path == 'xlm-roberta-base':
            ids_flota = self.tok.convert_tokens_to_ids(tokens)[:self.tok.model_max_length - 2]
            return ids_flota + [self.tok.sep_token_id, self.tok.cls_token_id]