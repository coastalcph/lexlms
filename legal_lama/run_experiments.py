# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass, field
from typing import Optional

import argparse
from legal_lama.batch_eval_KB_completion import main as run_evaluation
from legal_lama.batch_eval_KB_completion import load_file
from legal_lama.modules import build_model_by_name
import pprint
import statistics
from collections import defaultdict

from transformers import (
    HfArgumentParser,
    set_seed
)

set_seed(42)

uncased_models = {
    'nlpaueb/legal-bert-base-uncased',
    'nlpaueb/legal-bert-small-uncased',
    'zlucia/custom-legalbert',
    'lexlms/bert-base-uncased',
    'pile-of-law/legalbert-large-1.7M-2',
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    vocab_constraint: bool = field(
        default=False,
        metadata={
            "help": (
                "Constraint the predicted tokens to the set of available labels"
            )
        },
    )
    lowercase: bool = field(
        default=False,
        metadata={
            "help": (
                "Lowercase examples, even for cased models."
                "Uncased models will always lowercase examples."
            )
        },
    )


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    model_args
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")
    lowercase = True if model_args.model_name_or_path in uncased_models or model_args.lowercase else False
    constrained = 'constrained' if model_args.vocab_constraint else 'full'

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": None,
            "template": "",
            "bert_vocab_name": None,
            "batch_size": 16,
            "logdir": "output",
            "full_logdir": "/mnt/storage/ngarneau/legal-lama/output/results_fair_eval/{}/{}_{}".format(
                relation["relation"], model_args.model_name_or_path.replace("/", "_"),
                constrained
                ),
            "lowercase": lowercase,
            "max_sentence_length": 512,
            "threads": 1,
            "vocab_constraint": model_args.vocab_constraint
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            model, tokenizer, config = build_model_by_name(model_args)

        Precision1 = run_evaluation(args, model, tokenizer, config)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ - mean P@1: {}".format(mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "squad"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_ecthr_parameters(data_path_pre="data/"):
    relations = [{"relation": "ecthr_articles"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_contract_sections_parameters(data_path_pre="data/"):
    relations = [{"relation": "contract_sections"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_contract_types_parameters(data_path_pre="data/"):
    relations = [{"relation": "contract_types"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_canadian_sections_parameters(data_path_pre="data/"):
    relations = [{"relation": "canadian_crimes"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_cjeu_parameters(data_path_pre="data/"):
    relations = [{"relation": "cjeu_terms"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_echr_parameters(data_path_pre="data/"):
    relations = [{"relation": "ecthr_terms"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_us_crimes_parameters(data_path_pre="data/"):
    relations = [{"relation": "us_crimes"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_us_terms_parameters(data_path_pre="data/"):
    relations = [{"relation": "us_terms"}]
    data_path_pre += ""
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses()[0]

    print("1. ECTHR articles")
    parameters = get_ecthr_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("2. Contract sections")
    parameters = get_contract_sections_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("3. Contract types")
    parameters = get_contract_types_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("4. Canadian Sections")
    parameters = get_canadian_sections_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("5. CJEU Terms")
    parameters = get_cjeu_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("6. ECHR Terms")
    parameters = get_echr_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("7. US Crimes")
    parameters = get_us_crimes_parameters()
    run_experiments(*parameters, model_args=model_args)

    print("8. US Terms")
    parameters = get_us_terms_parameters()
    run_experiments(*parameters, model_args=model_args)


