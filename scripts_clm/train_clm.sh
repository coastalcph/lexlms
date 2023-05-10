export WANDB_PROJECT="lex-lm"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='lex-gpt-base'

python language_modeling/run_clm_stream.py \
    --model_name_or_path data/PLMs/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name data/the_legal_pile \
    --dataset_config_name uk-legislation,uk-courts-cases,indian-court-cases,us-contracts \
    --output_dir data/PLMs/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 5 \
    --max_steps 1000000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length