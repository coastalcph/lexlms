export WANDB_PROJECT="lex-lm"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PYTHONPATH=.

MODEL_MAX_LENGTH=1024
MODEL_PATH='lex-gpt/lex-gpt-base-v2'
BATCH_SIZE=8
ACCUMULATION_STEPS=1

python3 language_modeling/xla_spawn.py --num_cores=8 language_modeling/run_clm_stream.py \
    --model_name_or_path data/PLMs/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name data/the_legal_pile \
    --dataset_config_name all \
    --output_dir data/PLMs/${MODEL_PATH}-clm \
    --overwrite_output_dir \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 5 \
    --max_steps 50000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --line_by_line \
    --pad_to_max_length \
    --max_eval_samples 1000 \
    --local_dataset_loader 2
