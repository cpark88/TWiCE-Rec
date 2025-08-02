# python data_generator_v2.py \
# --sample_num=100000 \
# --default_next_token='<|n|>' \
# --default_query_token='<q>' \
# --min_len=5 \
# --max_len=50 \
# --data_name='amazon' \
# --data_path='one_model_sequence_v3_temp.json'


# python vocab_tokenizer.py \
# --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  \
# --subtoken_max_length=30 \
# --padding='y' \
# --data_name='amazon' \
# --default_next_token='<|n|>' \
# --default_query_token='<q>'  

DOMAIN_NAME="Grocery_and_Gourmet_Food"

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1234 train_reco.py \
--data_name 'amazon' \
--domain "$DOMAIN_NAME" \
--bf16 True \
--num_train_epochs 20 \
--per_device_train_batch_size 2048 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 20 \
--save_total_limit 1 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--logging_strategy "steps" \
--tf32 True \
--learning_rate 1e-3 \
--model_max_length 100 \
--neg_sample_type 'basic' \
--data_path "./amazon_dataset/amazon_"$DOMAIN_NAME"_sasrec_train_20250521.json" \
--output_dir "./output_dir/"$DOMAIN_NAME"/" | tee train_output_test_sasrec.log 




