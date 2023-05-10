export WANDB_PROJECT="lex-lm"
export PYTHONPATH=.

MODEL_PATH='lex-lm/lex-lm-tiny'
MODEL_MAX_LENGTH=128
TOTAL_STEPS=16
BATCH_SIZE=4

python language_modeling/run_mlm_stream.py \
       --model_name_or_path data/PLMs/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name data/the_legal_pile \
    --dataset_config_name all \
    --output_dir data/PLMs/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
    --logging_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 8 \
    --save_strategy steps \
    --save_steps 8 \
    --save_total_limit 2 \
    --max_steps ${TOTAL_STEPS} \
    --learning_rate 1e-4 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --max_eval_steps 4