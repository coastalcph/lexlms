# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# from .bert_connector import Bert
# from .elmo_connector import Elmo
# from .gpt_connector import GPT
# from .transformerxl_connector import TransformerXL
# from .roberta_connector import Roberta

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
import torch
LEXLMS_AUTH_KEY = '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'


def build_model_by_name(model_args):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=LEXLMS_AUTH_KEY,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=LEXLMS_AUTH_KEY,
        add_prefix_space=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=LEXLMS_AUTH_KEY,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer, config

