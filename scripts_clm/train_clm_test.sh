export WANDB_PROJECT="lex-lm"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='lex-gpt-base'

python language_modeling/run_clm_stream.py \
    --model_name_or_path data/PLMs/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name data/the_legal_pile \
    --dataset_config_name uk-court-cases \
    --output_dir data/PLMs/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
    --logging_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 8 \
    --save_strategy steps \
    --save_steps 16 \
    --save_total_limit 5 \
    --max_steps 16 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --max_seq_length 512 \
    --pad_to_max_length \
    --max_train_samples 16 \
    --max_eval_samples 4