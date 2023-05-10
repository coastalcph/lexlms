import warnings

import torch
import time
from transformers import AutoConfig, AutoModelForSequenceClassification
from modeling.hi_transformer import HiTransformerForSequenceClassification, HiTransformerConfig

warnings.filterwarnings("ignore")


def test_memory_usage(model, steps=100, batch_size=2, seq_length=1024):
    torch.cuda.reset_peak_memory_stats()
    model.to('cuda')
    input_ids = torch.randint(1, 30000, (batch_size, seq_length), dtype=torch.long).to('cuda')
    input_ids[:, :: 128] = 100
    labels = input_ids.clone()
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    start = time.time()
    for _ in range(steps):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    end = time.time()
    total_time = (end - start) / steps
    return torch.cuda.max_memory_allocated() / 1e9, total_time


def estimate_model_size():
    for CONFIG in ['tiny', 'small', 'base', 'large', 'xlarge']:
        roberta_config = AutoConfig.from_pretrained(f'data/PLMs/lex-lm/lex-lm-{CONFIG}')
        print('-' * 150)
        print(F'NUM LAYERS: {roberta_config.num_hidden_layers}\t'
              F'NUM HIDDEN: {roberta_config.hidden_size}\t'
              F'ATTENTION HEADS: {roberta_config.num_attention_heads}')
        print('-' * 150)

        # load dummy roberta model
        roberta_model = AutoModelForSequenceClassification.from_config(roberta_config)
        model_total_params = sum(p.numel() for p in roberta_model.roberta.parameters() if p.requires_grad)
        model_total_params = model_total_params / 1e6
        print(f'RoBERTa model has {model_total_params:.1f}M number of parameters.')

        # load dummy gpt model
        gpt_config = AutoConfig.from_pretrained(f'data/PLMs/lex-gpt/lex-gpt-{CONFIG}')
        gpt_model = AutoModelForSequenceClassification.from_config(gpt_config)
        model_total_params = sum(p.numel() for p in gpt_model.transformer.parameters() if p.requires_grad)
        model_total_params = model_total_params / 1e6
        print(f'GPT model has {model_total_params:.1f}M number of parameters.')

        # load dummy longformer model
        lf_config = AutoConfig.from_pretrained(f'data/PLMs/lex-longformer/lex-longformer-{CONFIG}')
        htf_model = AutoModelForSequenceClassification.from_config(lf_config)
        model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
        model_total_params = model_total_params / 1e6
        print(f'Longformer model has {model_total_params:.1f}M number of parameters.')

        # load dummy hi-transformer model
        htf_config = HiTransformerConfig.from_pretrained(f'data/PLMs/lex-hi-transformer/lex-hi-transformer-{CONFIG}')
        htf_model = HiTransformerForSequenceClassification.from_config(htf_config)
        model_total_params = sum(p.numel() for p in htf_model.hi_transformer.parameters() if p.requires_grad)
        model_total_params = model_total_params / 1e6
        print(f'Hi-transformer model has {model_total_params:.1f}M number of parameters.')


if __name__ == '__main__':
    estimate_model_size()


'''
------------------------------------------------------------------------------------------------------------------------------------------------------
NUM LAYERS: 4	NUM HIDDEN: 256	ATTENTION HEADS: 4
------------------------------------------------------------------------------------------------------------------------------------------------------
RoBERTa model has 15.0M number of parameters.
GPT model has 16.2M number of parameters.
Longformer model has 16.7M number of parameters.
Hi-transformer model has 16.0M number of parameters.
------------------------------------------------------------------------------------------------------------------------------------------------------
NUM LAYERS: 8	NUM HIDDEN: 384	ATTENTION HEADS: 6
------------------------------------------------------------------------------------------------------------------------------------------------------
RoBERTa model has 28.9M number of parameters.
GPT model has 33.8M number of parameters.
Longformer model has 33.8M number of parameters.
Hi-transformer model has 33.5M number of parameters.
------------------------------------------------------------------------------------------------------------------------------------------------------
NUM LAYERS: 12	NUM HIDDEN: 768	ATTENTION HEADS: 12
------------------------------------------------------------------------------------------------------------------------------------------------------
RoBERTa model has 123.9M number of parameters.
GPT model has 124.2M number of parameters.
Longformer model has 129.0M number of parameters.
Hi-transformer model has 151.9M number of parameters.
------------------------------------------------------------------------------------------------------------------------------------------------------
NUM LAYERS: 24	NUM HIDDEN: 1024	ATTENTION HEADS: 16
------------------------------------------------------------------------------------------------------------------------------------------------------
RoBERTa model has 354.0M number of parameters.
GPT model has 354.6M number of parameters.
Longformer model has 433.3M number of parameters.
Hi-transformer model has 429.3M number of parameters.
'''
