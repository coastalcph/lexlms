MODEL_NAME='lexlms/roberta-base'
BATCH_SIZE=16
TASK='eurlex'
export CUDA_VISIBLE_DEVICES=5

python lex-glue/experiments/eurlex.py \
  --model_name_or_path ${MODEL_NAME} \
  --output_dir data/lexglue_logs/${TASK}/${MODEL_NAME} \
  --max_seq_length 2048 \
  --do_train \
  --do_eval \
  --do_pred \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model micro-f1 \
  --greater_is_better True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 5 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --seed 42 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --fp16 \
  --fp16_full_eval \
  --lr_scheduler_type cosine
