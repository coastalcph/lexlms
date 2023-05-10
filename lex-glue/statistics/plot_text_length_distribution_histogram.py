from datasets import load_dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tqdm
from transformers import AutoTokenizer
from data import AUTH_KEY
import re


TOKENIZER = AutoTokenizer.from_pretrained('lexlms/roberta-base', use_auth_token=AUTH_KEY)

# for subset_name in ['ecthr_a', 'scotus', 'eurlex', 'ildc', 'case_hold', 'ledgar', 'contractnli_a',
#                     'uklex', 'unfair_tos', 'ecthr_arguments', 'contractnli_b']:

for subset_name in ['ledgar']:
    dataset = load_dataset('lexlms/lex_glue_v2', name=subset_name, use_auth_token=AUTH_KEY)

    text_length = []
    if subset_name == 'case_hold':
        text_column = 'contexts'
    elif subset_name in ['contractnli_a', 'contractnli_b']:
        text_column = 'premise'
    else:
        text_column = 'text'

    for idx, document in tqdm.tqdm(enumerate(dataset['train'][text_column])):
        if subset_name in ['ecthr_a', 'ecthr_arguments']:
            if isinstance(document, list):
                document = ' '.join(document)
        elif subset_name == 'case_hold':
            document = document[0] + ' ' + dataset['train']['endings'][idx][0]
        elif subset_name in ['contractnli_a', 'contractnli_b']:
            document += ' ' + dataset['train']['hypothesis'][idx]
        document = re.sub(r'\n+', '\n', re.sub(r'( |\t)+', ' ', document))
        text_length.append(len(TOKENIZER.tokenize(document)))

    text_length = sorted(text_length)
    semi_last_percentile_text = text_length[int(len(text_length) * 0.9)]
    last_percentile_text = text_length[int(len(text_length) * 0.95)]

    plt.hist(text_length, bins=250, range=(0, 10000), alpha=0.5, edgecolor='k')
    plt.legend(loc='upper right')
    plt.xlabel('Number of sub-word units (tokens)')
    plt.ylabel('Number of documents')
    plt.title(f'90% percentile: {semi_last_percentile_text} sub-word units (tokens)\n'
              f'95% percentile: {last_percentile_text} sub-word units (tokens)')
    plt.savefig(f'plots/{subset_name}.png', bbox_inches="tight")
    plt.clf()



