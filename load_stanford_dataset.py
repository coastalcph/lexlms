import tqdm
from datasets import load_dataset

DATASET_NAMES = ['r_legaladvice', 'atticus_contracts',
                 'federal_register', 'bva_opinions', 'us_bills', 'cc_casebooks', 'tos', 'nlrb_decisions',
                 'scotus_oral_arguments', 'cfr', 'state_codes', 'scotus_filings', 'bar_exam_outlines', 'edgar',
                 'cfpb_creditcard_contracts', 'constitutions', 'oig', 'olc_memos', 'uscode', 'ftc_advisory_opinions',
                 'echr', 'eurlex', 'tax_rulings', 'fre', 'frcp', 'canadian_decisions',
                 'eoir', 'dol_ecab', 'courtlistener_opinions', 'courtlistener_docket_entry_documents']

for dataset_name in DATASET_NAMES:
    total_documents = 0
    total_tokens = 0
    for subset in ['train', 'validation']:
        documents = 0
        tokens = 0
        dataset = load_dataset('pile-of-law/pile-of-law', dataset_name, split=subset, streaming=True)
        for example in tqdm.tqdm(iter(dataset)):
            documents += 1
            tokens += len(example['text'].split())
        total_documents += documents
        total_tokens += tokens
        print(f'{dataset_name.upper():>20}[{subset:>10}]: {documents:>10} {tokens:>10}')
    print(f'{dataset_name.upper():>20}[total]: {total_documents:>10} {total_tokens:>10}')
