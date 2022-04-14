#!/bin/sh
python3 ../cli.py --method pet --task_name idiom-detection --pattern_ids 3 --data_dir ../semeval_data/ --model_type bert --model_name_or_path bert-base-multilingual-cased --output_dir ../semeval/pet-1000-p3 --do_train --do_eval --train_examples 1000 --unlabeled_examples 3000 --split_examples_evenly --pet_per_gpu_train_batch_size 1 --pet_per_gpu_unlabeled_batch_size 1 --pet_gradient_accumulation_steps 16 --pet_max_steps 250 --lm_training --sc_per_gpu_train_batch_size 1 --sc_per_gpu_unlabeled_batch_size 1 --sc_gradient_accumulation_steps 16 --sc_max_steps 5000 --pet_max_seq_lengt 512 --sc_max_seq_length 512