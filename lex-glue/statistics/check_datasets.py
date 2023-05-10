from datasets import load_dataset
from data import AUTH_KEY

# for subset_name in ['ecthr_a', 'scotus', 'eurlex', 'ildc', 'case_hold', 'ledgar', 'contractnli_a']:
for subset_name in ['contractnli_a']:
    dataset = load_dataset('lexlms/lex_glue_v2', name=subset_name, use_auth_token=AUTH_KEY)
    for subset in ['train', 'validation', 'test']:
        print(f'{subset_name.upper()}[{subset.upper():>10}]: {len(dataset[subset])}')
#
# name_labels = [dataset['train'].features['labels'].feature.names[label] for doc_labels in dataset['train']['labels'] for
#                label in doc_labels]
# id_labels = [label for doc_labels in dataset['train']['labels'] for label in doc_labels]
#
# from collections import Counter
#
# counts = Counter(name_labels)
# top_100_names = [label for label, count in counts.most_common(n=100)]
#
# counts = Counter(id_labels)
# top_100_ids = [label for label, count in counts.most_common(n=100)]
#
# print()
#
# labels = [119, 120, 114, 90, 28, 29, 30, 82, 87, 8, 44, 31, 33, 94, 22, 14, 52, 91, 92, 13, 89, 86, 118, 93, 12, 68, 83,
#           98, 11, 7, 32, 115, 96, 79, 116, 106, 81, 75, 117, 112, 59, 6, 77, 95, 72, 108, 60, 99, 74, 24, 27, 34, 58,
#           66, 84, 61, 16, 107, 20, 43, 97, 105, 76, 67, 80, 57, 63, 37, 36, 85, 5, 109, 69, 38, 78, 39, 49, 23, 42, 100,
#           17, 70, 9, 51, 113, 103, 102, 110, 0, 41, 111, 101, 35, 64, 10, 121, 21, 26, 71, 122]
