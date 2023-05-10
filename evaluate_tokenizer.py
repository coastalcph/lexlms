from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from compute_sampling_ratio import compute_sampling_rates
import os
from data import DATA_DIR

CUSTOM_TOK_FOLDER = os.path.join(DATA_DIR, 'PLMs/lex-lm')


def main():

    _, probabilities, sampling_rates = compute_sampling_rates()

    # load datasets
    datasets = []
    print(sampling_rates.keys())
    for dataset_name in sampling_rates.keys():
        datasets.append(load_dataset(os.path.join(DATA_DIR, 'lexfiles_processed'), f"{dataset_name}-chunks_512",
                                     split='train', streaming=True))

    # interleave datasets with sampling rates
    dataset = interleave_datasets(datasets, probabilities=list(sampling_rates.values()),
                                  stopping_strategy='all_exhausted')

    tokenizer = AutoTokenizer.from_pretrained(f'{CUSTOM_TOK_FOLDER}-large')

    test_samples = dataset.take(500)
    for example in test_samples:
        sample_text = ' '.join(example['text'].split(' ')[:500])
        print(sample_text)
        print('-' * 150)
        print(tokenizer.tokenize(sample_text))
        print('-' * 150)


if __name__ == "__main__":
    main()
