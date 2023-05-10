export WANDB_PROJECT="lex-lm"
export PYTHONPATH=.

MODEL_PATH='lex-longformer-base'
MODEL_MAX_LENGTH=4096
TOTAL_STEPS=64000
ACCUMULATION_STEPS=2
BATCH_SIZE=8

python language_modeling/run_mlm_stream.py \
    --model_name_or_path data/PLMs/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name data/the_legal_pile_original \
    --dataset_config_name all \
    --output_dir data/PLMs/${MODEL_PATH}-mlm \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 20000 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 5 \
    --max_steps 64000 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability 0.20 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --line_by_line \
    --pad_to_max_length \
    --max_eval_samples 1000
