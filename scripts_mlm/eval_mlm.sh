export WANDB_PROJECT="lex-lm-eval"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

MODEL_MAX_LENGTH=512
MODEL_PATH='lexlms/roberta-base-cased'
BATCH_SIZE=32
DATASET_NAME='the_legal_pile'

python language_modeling/eval_mlm_stream.py \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir data/PLMs/${MODEL_PATH}-${DATASET_NAME}-mlm-eval \
    --overwrite_output_dir \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --line_by_line \
    --pad_to_max_length
