# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
import torch
from legal_lama.modules import build_model_by_name
import legal_lama.utils as utils
from legal_lama.utils import print_sentence_predictions, load_vocab
import legal_lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import legal_lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import legal_lama.evaluation_metrics as metrics
import time, sys
import numpy as np


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["text"]).split())
    ):
        masked_sentences = sample["text"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg


def run_thread(arguments):

    msg = ""

    if len(arguments["filtered_log_probs"]) != len(arguments["label_index"]):
        import pdb; pdb.set_trace()
    


    # 1. compute the ranking metrics on the filtered log_probs tensor
    sample_MRR, sample_P, experiment_result, return_msg = metrics.get_ranking(
        arguments["filtered_log_probs"],
        arguments["masked_indices"],
        arguments["vocab"],
        label_index=arguments["label_index"],
        index_list=arguments["index_list"],
        print_generation=arguments["interactive"],
        topk=10000,
        vocab_subset_indices=arguments['vocab_subset_indices'],
        sample=arguments['sample']
    )
    msg += "\n" + return_msg

    sample_perplexity = 0.0

    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg, arguments['label'], arguments['top_label']


def lowercase_samples(samples, use_negated_probes=False):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["text"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["text"] = lower_masked_sentences

        if "negated" in sample and use_negated_probes:
            for sentence in sample["negated"]:
                sentence = sentence.lower()
                sentence = sentence.replace(base.MASK.lower(), base.MASK)
                lower_masked_sentences.append(sentence)
            sample["negated"] = lower_masked_sentences

        new_samples.append(sample)
    return new_samples


def filter_samples(model, tokenizer, samples, vocab_subset, max_sentence_length, template):
    msg = ""
    new_samples = []
    samples_exluded = 0
    for sample in samples:
        excluded = False
        if "label" in sample:
            sample['label'] = sample['label'].lower()

            if sample["text"].count('<mask>') > 1:
                continue

            if len(tokenizer(sample["text"])['input_ids']) > max_sentence_length:
                continue

            obj_label_ids = tokenizer.encode(sample["label"], add_special_tokens=False)

            if obj_label_ids:
                recostructed_word = tokenizer.decode(obj_label_ids).strip()
            else:
                recostructed_word = None

            if obj_label_ids is None:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            else:
                new_samples.append(sample)
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg


def main(args, model, tokenizer, config, data):

    msg = ""
    model_name = config._name_or_path

    # initialize logging
    if args.full_logdir:
        log_directory = args.full_logdir
    else:
        log_directory = create_logdir_with_timestamp(args.logdir, model_name)
    logger = init_logging(log_directory)
    msg += "model name: {}\n".format(model_name)

    # deal with vocab subset
    vocab_subset = None
    index_list = None
    msg += "args: {}\n".format(args)


    logger.info("\n" + msg + "\n")

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        json.dump(vars(args), outfile)

    # Mean reciprocal rank
    MRR = 0.0

    # Precision at (default 10)
    Precision = 0.0
    Precision1 = 0.0

    # stats per obj_label
    stats_per_label = defaultdict(lambda: defaultdict(list))
    examples_scores = list()

    print(len(data))

    if args.lowercase:
        # lowercase all samples
        logger.info("lowercasing all samples...")
        all_samples = lowercase_samples(
            data, use_negated_probes=False
        )
    else:
        # keep samples as they are
        all_samples = data

    if args.vocab_constraint:  # Constrain on the set of possible labels
        labels_metadata = defaultdict(list)
        vocab_subset = {d['label'].lower() for d in data}
        for label in vocab_subset:
            indices = tokenizer(label, add_special_tokens=False)['input_ids']
            length = len(indices)
            labels_metadata[length].append({
                "label": label,
                "length": length,
                "indices": indices
            })
        longest_label = max(labels_metadata.keys())
        shortest_label = min(labels_metadata.keys())
        vocab_subset_indices = tokenizer(list(vocab_subset), add_special_tokens=False)['input_ids']
        vocab_subset_indices = {char for word in vocab_subset_indices for char in word}
    else:
        vocab_subset_indices = None

    all_samples, ret_msg = filter_samples(
        model, tokenizer, data, vocab_subset, args.max_sentence_length, args.template
    )

    logger.info("\n" + ret_msg + "\n")

    print(len(all_samples))

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    samples_batches, sentences_batches, ret_msg = batchify(all_samples, args.batch_size)
    logger.info("\n" + ret_msg + "\n")

    list_of_results = []

    for i in tqdm(range(len(samples_batches))):

        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        for sample, sentence in zip(samples_b, sentences_b):
            real_label = sample['label']
            top_label = sample['category']
            labels_probabilities = dict()
            for length, labels in labels_metadata.items():
                new_mask = " ".join(['<mask>'] * length)
                new_sample = sample['text'][0].replace('<mask>', new_mask)
                sentence_masked = new_sample.replace("<mask>", tokenizer.mask_token)
                inputs = tokenizer(sentence_masked, return_tensors="pt", padding=True, truncation=True, max_length=args.max_sentence_length)

                if torch.cuda.is_available():
                    inputs = inputs.to(model.device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0]

                masked_indices_list = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
                masked_indices = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero().squeeze(1)
                for label_metadata in labels:
                    label_indices = label_metadata['indices']
                    probs = list()
                    for l_index, l_m in zip(label_indices, masked_indices):
                        soft_probs = torch.nn.functional.softmax(logits, dim=1)
                        prob = soft_probs[l_m, l_index]
                        probs.append(float(prob))
                    labels_probabilities[label_metadata['label']] = np.mean(probs)  # mean
            sorted_labels = sorted(labels_probabilities.items(), key=lambda x: x[1], reverse=True)  # When using perplexity, do no reverse

            for rank, (label, prob) in enumerate(sorted_labels):
                rank = rank + 1
                if label == real_label:
                    mrr = 1/rank
                    if rank == 1:
                        p1 = 1.0
                    else:
                        p1 = 0.0
            stats_per_label[top_label]['MRR_results'].append(mrr)
            stats_per_label[top_label]['Precision1_results'].append(p1)
            sample['mrr'] = mrr
            examples_scores.append(sample)

    for label, results in stats_per_label.items():
        stats_per_label[label]['MRR'] = sum(results['MRR_results']) / len(results['MRR_results'])
        stats_per_label[label]['Precision1'] = sum(results['Precision1_results']) / len(results['Precision1_results'])

    stats_per_label = dict(stats_per_label)

    macro_MRR = np.mean([np.mean(v['MRR_results']) for _, v in stats_per_label.items()])
    macro_P1 = np.mean([np.mean(v['Precision1_results']) for _, v in stats_per_label.items()])

    msg = "all_samples: {}\n".format(len(all_samples))
    msg += "global MRR: {}\n".format(macro_MRR)
    msg += "global Precision at 1: {}\n".format(macro_P1)

    logger.info("\n" + msg + "\n")
    print("\n" + msg + "\n")

    # dump pickle with the result of the experiment
    all_results = dict(
        stats_per_label=stats_per_label,
        macro_MRR=macro_MRR,
        macro_P1=macro_P1
    )
    with open("{}/result.pkl".format(log_directory), "wb") as f:
        pickle.dump(all_results, f)

    json.dump(examples_scores, open("{}/samples_scores.json".format(log_directory), "w"), indent=True)

    return Precision1


if __name__ == "__main__":
    parser = options.get_eval_KB_completion_parser()
    args = options.parse_args(parser)
    main(args)
