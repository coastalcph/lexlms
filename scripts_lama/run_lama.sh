#!/bin/bash

export WANDB_PROJECT="lex-lm"
export PYTHONPATH=.

models=(
    'roberta-base'
    'roberta-large'
    'nlpaueb/legal-bert-base-uncased'
    'zlucia/custom-legalbert'
    'pile-of-law/legalbert-large-1.7M-2'
    'lexlms/roberta-base'
    'lexlms/roberta-large'
    )
for model in "${models[@]}"
do
    echo "$model"
    python -m legal_lama.run_experiments --model_name_or_path ${model} --vocab_constraint true
done