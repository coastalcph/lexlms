import numpy
import tqdm
from tokenizers import models, pre_tokenizers, decoders, processors, trainers, normalizers

from tokenizers import Tokenizer
from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig
from compute_sampling_ratio import compute_sampling_rates
import os
import argparse
from data import DATA_DIR

MODEL_FOLDER = os.path.join(DATA_DIR, 'PLMs/lex-lm')


def batch_iterator(dataset):
    for example in dataset:
        yield example['text']


def train_tokenizer():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--vocab_size', default=64000, type=int),
    parser.add_argument('--cased', default=False, type=bool,
                        help='Cased or Uncased tokenizer')
    parser.add_argument('--max_examples', default=10e6, type=int)
    parser.add_argument('--overwrite_dir', default=True, type=bool)
    config = parser.parse_args()

    print('Configuration')
    print('--------------')
    print(f'Vocab-Size: {config.vocab_size}')
    print(f'Cased: {bool(config.cased)}')
    print('--------------')

    # Configure tokenizer
    # BPE tokenizer
    backend_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Unicode and whitespace normalization
    backend_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKD(), normalizers.BertNormalizer(lowercase=not config.cased)])
    # Whitespace pre-tokenization followed by ByteLevel
    backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.BertPreTokenizer(), pre_tokenizers.ByteLevel(add_prefix_space=True)])
    # ByteLevel decoding
    backend_tokenizer.decoder = decoders.ByteLevel()
    # RoBERTa post-processing
    backend_tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1),
                                                                    add_prefix_space=True, trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )

    # Load sampling rates
    _, probabilities, sampling_rates = compute_sampling_rates()

    # Load datasets
    datasets = []
    print(sampling_rates.keys())
    for dataset_name in sampling_rates.keys():
        datasets.append(load_dataset('lexlms/lexfiles_processed',
                                     dataset_name,
                                     split='train',
                                     streaming=True))

    # Interleave datasets with pre-defined sampling rates
    dataset = interleave_datasets(datasets, probabilities=list(sampling_rates.values()),
                                  stopping_strategy='all_exhausted')

    train_dataset = dataset.take(int(config.max_examples))
    dev_dataset = dataset.skip(int(config.max_examples))

    if config.cased:
        casing = 'cased'
    else:
        casing = 'uncased'

    if not os.path.exists(f'{MODEL_FOLDER}-large-{casing}') or config.overwrite_dir:
        # Train tokenizer
        backend_tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator(train_dataset),
                                              length=int(config.max_examples))

        # Save tokenizer
        new_roberta_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            model_max_length=512,
            # padding_side="Set me if you want",
            # truncation_side="Set me if you want",
            # model_input_names="Set me if you want",
            bos_token='<s>',
            eos_token='</s>',
            unk_token='<unk>',
            sep_token='</s>',
            pad_token='<pad>',
            cls_token='<s>',
            mask_token='<mask>',
        )

        # Save resources
        for config_size in ['base', 'large']:
            # Load config and save in new folder
            config = AutoConfig.from_pretrained(f'{MODEL_FOLDER}-{config_size}')
            config.save_pretrained(f'{MODEL_FOLDER}-{config_size}-{casing}')
            # Save tokenizer
            new_roberta_tokenizer.save_pretrained(f'{MODEL_FOLDER}-{config_size}-{casing}')
            print(f'Trained a {casing} BPE tokenizer with a vocabulary of {config.vocab_size} sub-words successfully!')

    print(f'Evaluating tokenizers!')
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_FOLDER}-large-{casing}')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Test tokenizers with 1000 examples
    test_samples = dev_dataset.take(1000)
    custom_token_frag = []
    roberta_token_frag = []
    with open(f'tokenization-examples-{casing}.txt', 'w') as out_file:
        for example in tqdm.tqdm(test_samples):
            out_file.write('-' * 150 + '\n')
            out_file.write('ORIGINAL TEXT\n')
            out_file.write('-' * 150 + '\n')
            out_file.write(example['text'] + '\n')
            out_file.write('-' * 150 + '\n')
            out_file.write('CUSTOM TOKENIZER\n')
            out_file.write('-' * 150 + '\n')
            custom_tokens = tokenizer.tokenize(example['text'])
            custom_token_frag.append(len(custom_tokens) / len(example['text'].split()))
            out_file.write(str(custom_tokens) + '\n')
            out_file.write('-' * 150 + '\n')
            out_file.write('ROBERTA TOKENIZER\n')
            out_file.write('-' * 150 + '\n')
            roberta_tokens = roberta_tokenizer.tokenize(example['text'])
            roberta_token_frag.append(len(roberta_tokens) / len(example['text'].split()))
            out_file.write(str(roberta_tokens) + '\n')
        out_file.write(f'Custom  tokenizer fragmentation rate: {numpy.mean(custom_token_frag):.2f}.\n')
        out_file.write(f'Roberta tokenizer fragmentation rate: {numpy.mean(roberta_token_frag):.2f}.\n')
        print(f'Custom  tokenizer fragmentation rate: {numpy.mean(custom_token_frag):.2f}.')
        print(f'Roberta tokenizer fragmentation rate: {numpy.mean(roberta_token_frag):.2f}.')


if __name__ == "__main__":
    train_tokenizer()
