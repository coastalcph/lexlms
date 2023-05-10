export WANDB_PROJECT="lex-lm"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='lex-lm-large-cased-v2'
BATCH_SIZE=8
ACCUMULATION_STEPS=4

python3 language_modeling/xla_spawn.py --num_cores=8 language_modeling/run_mlm_stream.py \
    --model_name_or_path data/PLMs/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name lexlms/the_legal_pile_preprocessed \
    --dataset_config_name all \
    --output_dir data/PLMs/${MODEL_PATH}-mlm \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 5 \
    --max_steps 1000000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability 0.30 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --line_by_line \
    --pad_to_max_length \
    --max_eval_samples 10000 \
    --freeze_model_encoder true



