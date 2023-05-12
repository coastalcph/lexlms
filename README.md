# LeXFiles and LegalLAMA: Facilitating English Multinational Legal Language Model Development :balance_scale: :woman_judge: :world_map:


## Introduction

### Citation

[*Ilias Chalkidis\*, Nicolas Garneau\*, Catalina E.C. Goanta, Daniel Martin Katz, and Anders Søgaard.*
*LeXFiles and LegalLAMA: Facilitating English Multinational Legal Language Model Development.*
*2022. In the Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics. Toronto, Canada.*](https://aclanthology.org/xxx/)
```
@inproceedings{chalkidis-garneau-etal-2023-lexlms,
    title = {{LeXFiles and LegalLAMA: Facilitating English Multinational Legal Language Model Development}},
    author = "Chalkidis*, Ilias and 
              Garneau*, Nicolas and
              Goanta, Catalina and 
              Katz, Daniel Martin and 
              Søgaard, Anders",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/xxx",
}
```

## The LexFiles Corpus

The LeXFiles is a new diverse English multinational legal corpus that we created including 11 distinct sub-corpora that cover legislation and case law from 6 primarily English-speaking legal systems (EU, CoE, Canada, US, UK, India).
The corpus contains approx. 19 billion tokens. In comparison, the "Pile of Law" corpus released by Hendersons et al. (2022) comprises 32 billion in total, where the majority (26/30) of sub-corpora come from the United States of America (USA), hence the corpus as a whole is biased towards the US legal system in general, and the federal or state jurisdiction in particular, to a significant extent.

### Usage

```python
from datasets import load_dataset
dataset = load_dataset('lexlms/lexfiles', 'eu-legislation')
```


### Dataset Specifications


#### The LexFiles
| Corpus                            | Corpus alias         | Documents | Tokens | Pct.   | Sampl. (a=0.5) | Sampl. (a=0.2) |
|-----------------------------------|----------------------|-----------|--------|--------|----------------|----------------|
| :eu: EU Legislation               | `eu-legislation`     | 93.7K     | 233.7M | 1.2%   | 5.0%           | 8.0%           |
| :eu: EU Court Decisions           | `eu-court-cases`     | 29.8K     | 178.5M | 0.9%   | 4.3%           | 7.6%           |
| :eu: ECtHR Decisions              | `ecthr-cases`        | 12.5K     | 78.5M  | 0.4%   | 2.9%           | 6.5%           |
| :uk: UK Legislation               | `uk-legislation`     | 52.5K     | 143.6M | 0.7%   | 3.9%           | 7.3%           |
| :uk: UK Court Decisions           | `uk-court-cases`     | 47K       | 368.4M | 1.9%   | 6.2%           | 8.8%           |
| :india: Indian Court Decisions    | `indian-court-cases` | 34.8K     | 111.6M | 0.6%   | 3.4%           | 6.9%           |
| :canada: Canadian Legislation     | `canada-legislation` | 6K        | 33.5M  | 0.2%   | 1.9%           | 5.5%           |
| :canada: Canadian Court Decisions | `canadian_decisions` | 11.3K     | 33.1M  | 0.2%   | 1.8%           | 5.4%           |
| :us: U.S. Court Decisions [1]     | `court-listener`     | 4.6M      | 11.4B  | 59.2%  | 34.7%          | 17.5%          |
| :us: U.S. Legislation             | `us-legislation`     | 518       | 1.4B   | 7.4%   | 12.3%          | 11.5%          |
| :us: U.S. Contracts               | `us-contracts`       | 622K      | 5.3B   | 27.3%  | 23.6%          | 15.0%          |
| Total                             | `lexlms/lexfiles`    | 5.8M      | 18.8B  | 100%   | 100%           | 100%           |

[1] We consider only U.S. Court Decisions from 1965 onwards (cf. post Civil Rights Act), as a hard threshold for cases relying on severely out-dated and in many cases harmful law standards. The rest of the corpora include more recent documents.

[2] Sampling (Sampl.) ratios are computed following the exponential sampling introduced by Lample et al. (2019).

Additional corpora not considered for pre-training, since they do not represent factual legal knowledge.

| Corpus                                 | Corpus alias           | Documents | Tokens | 
|----------------------------------------|------------------------|-----------|--------|
| :world_map: Legal web pages from C4    | `legal-c4`             | 284K      | 340M   |


## LegalLAMA: Legal Language Model Analysis

LegalLAMA is a diverse probing benchmark suite comprising 8 sub-tasks that aims to assess the acquaintance of legal knowledge that PLMs acquired in pre-training.

### Dataset Specifications

| Corpus                                   | Corpus alias         | Examples  | Avg. Tokens | Labels |
|------------------------------------------|----------------------|-----------|-------------|--------|
| :canada: Criminal Code Sections (Canada) | `canadian_sections`  | 321       | 72          | 144    |
| :eu: Legal Terminology (EU)              | `cjeu_term`          | 2,127     | 164         | 23     |
| :us: Contractual Section Titles (US)     | `contract_sections`  | 1,527     | 85          | 20     |
| :us: Contract Types (US)                 | `contract_types`     | 1,089     | 150         | 15     |
| :eu: ECHR Articles (CoE)                 | `ecthr_articles`     | 5,072     | 69          | 13     |
| :eu: Legal Terminology (CoE)             | `ecthr_terms`        | 6,803     | 97          | 250    |
| :us: Crime Charges (US)                  | `us_crimes`          | 4,518     | 118         | 59     |
| :us: Legal Terminology (US)              | `us_terms`           | 5,829     | 308         | 7      |

### Evaluating PLMs

We considered the following PLMs in our experimentations;
- RoBERTa
- LegalBERT
- CaseLawBERT
- PoL-BERT
- LexLM

To evaluate these PLMs on LegalLAMA, run the following script:
```
sh scripts_lama/run_lama.sh
```


## LexLMs - Pre-trained Language Models (PLMs)

We release 2 new legal-oriented PLMs, dubbed LexLMs, warm-started from the RoBERTa models, and further pre-trained on the "LeXFiles" corpuss for 1M additional steps.

| Model Name       | Alias                                                                     | Layers | Hidden Units | Attention Heads | Parameters |
|------------------|---------------------------------------------------------------------------|--------|--------------|-----------------|------------|
| Lex-LM (Base)   | [`lexlms/legal-roberta-base`](https://huggingface.co/lexlms/legal-roberta-base) | 12     | 768    | 12              | 123.9M     |
| Lex-LM (Large)  | [`lexlms/legal-roberta-large`](https://huggingface.co/lexlms/legal-roberta-large)     |   24   | 1024   | 16              | 354.0M     |

### Usage

You can load any model with the standard HF AutoModel code.

```python 
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("lexlms/legal-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("lexlms/legal-roberta-base")
```


## Train Language Models

### Train Tokenizer

Initially, we train a custom BPE tokenizer:

```
python train_tokenizer.py
```

### Train Language Model

Then, we train an LM from scratch with Masked Language Modelling (MLM):

```
sh scripts_mlm/train_lm.sh
```

Model type (architecture, size) and other training specifications can be modified in the script:

```
MODEL_PATH='lexlm-large'
MODEL_MAX_LENGTH=512
TOTAL_STEPS=1000000
BATCH_SIZE=512
```

## Fine-tune Language Models

### Fine-tune on LexGLUE tasks

For example to fine-tune a PLM on EURLEX, you have to run:

```
sh lex-glue/scripts/run_eurlex.sh
```
PLM and other training specifications can be modified in the script:

```
MODEL_PATH='lexlms/legal-roberta-large'
