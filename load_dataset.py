import tqdm
from datasets import load_dataset

DATASET_NAMES = ['eu-legislation', 'eu-court-cases', 'ecthr-cases',
                 'uk-legislation', 'uk-court-cases', 'indian-court-cases',
                 'canadian-legislation', 'canadian-court-cases',
                 'us-contracts', 'us-court-cases', 'us-legislation']


for dataset_name in DATASET_NAMES:
    total_documents = 0
    total_tokens = 0
    for subset in ['train', 'test']:
        documents = 0
        tokens = 0
        dataset = load_dataset('lexlms/lexfiles', dataset_name, split=subset,  streaming=True)
        for example in tqdm.tqdm(iter(dataset)):
            documents += 1
            tokens += len(example['text'].split())
        total_documents += documents
        total_tokens += tokens
        print(f'{dataset_name.upper():>20}[{subset:>10}]: {documents:>10} {tokens:>10}')
    print(f'{dataset_name.upper():>20}[total]: {total_documents:>10} {total_tokens:>10}')


'''
      EU-LEGISLATION[     train]:      79164  154457015
      EU-LEGISLATION[validation]:       6829   28932100
      EU-LEGISLATION[      test]:       7675   50328666
      EU-LEGISLATION[     total]:      93668  233717781
--------------------------------------------------------
      EU-COURT-CASES[     train]:      24333  133999439
      EU-COURT-CASES[validation]:       3171   25274096
      EU-COURT-CASES[      test]:       2325   19177441
      EU-COURT-CASES[     total]:      29829  178450976
--------------------------------------------------------
         ECTHR-CASES[     train]:      10528   65500665
         ECTHR-CASES[validation]:       1000    6406440
         ECTHR-CASES[      test]:       1000    6618089
         ECTHR-CASES[     total]:      12528   78525194
--------------------------------------------------------
        US-CONTRACTS[     train]:     581560 4843667680
        US-CONTRACTS[validation]:      20000  208732990
        US-CONTRACTS[      test]:      20000  208799884
        US-CONTRACTS[     total]:     621560 5261200554
--------------------------------------------------------
            LEGAL-C4[     train]:     263937  316349514
            LEGAL-C4[validation]:      10000   11425461
            LEGAL-C4[      test]:      10000   12266921
            LEGAL-C4[     total]:     283937  340041896
--------------------------------------------------------
      UK-LEGISLATION[     train]:      44514  114585060
      UK-LEGISLATION[validation]:       4000   13712374
      UK-LEGISLATION[      test]:       4000   15326489
      UK-LEGISLATION[     total]:      52514  143623923
--------------------------------------------------------
      UK-COURT-CASES[     train]:      39040  295205794
      UK-COURT-CASES[validation]:       4000   35708430
      UK-COURT-CASES[      test]:       4000   37519140
      UK-COURT-CASES[     total]:      47040  368433364
--------------------------------------------------------
  INDIAN-COURT-CASES[     train]:      28816   92590416
  INDIAN-COURT-CASES[validation]:       3000    9301351
  INDIAN-COURT-CASES[      test]:       3000    9679726
  INDIAN-COURT-CASES[     total]:      34816  111571493
--------------------------------------------------------
      US-COURT-CASES[     train]:    4430756 11182535887
      US-COURT-CASES[validation]:     100000   59492096
      US-COURT-CASES[      test]:     100000  164914363
      US-COURT-CASES[     total]:    4630756 11406942346
'''
