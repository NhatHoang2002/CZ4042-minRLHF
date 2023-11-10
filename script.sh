#!/bin/bash


accelerate launch --config_file="accelerate_config/sft_config.yaml" sft_training.py \
                --model_name_or_path="gpt2-medium" \
                --output_dir="saved/gpt2_medium_uc_sft" --dataset_names="uc"
accelerate launch --config_file="accelerate_config/dpo_config.yaml" dpo_training.py \
                --model_name_or_path="saved/gpt2_medium_uc_sft" \
                --output_dir="saved/gpt2_medium_ufb_dpo" --dataset_names="ufb"
python merge_peft_adapter.py --base_model_name="gpt2-medium" --adapter_model_name="saved/gpt2_medium_ufb_dpo" --output_name="gpt2_medium_uc_ufb_merged"



# accelerate launch --config_file="accelerate_config/sft_config.yaml" sft_training.py \
#                 --model_name_or_path="gpt2-large" \
#                 --output_dir="saved/gpt2_large_uc_hh_sft" --dataset_names="uc,hh"
# accelerate launch --config_file="accelerate_config/dpo_config.yaml" dpo_training.py \
#                 --model_name_or_path="saved/gpt2_large_uc_hh_sft" \
#                 --output_dir="saved/gpt2_large_ufb_dpo" --dataset_names="ufb"
# python merge_peft_adapter.py --base_model_name="gpt2-large" --adapter_model_name="saved/gpt2_large_ufb_dpo" --output_name="gpt2_large_uc_ufb_merged"



# accelerate launch reward_modeling.py --model_name_or_path="saved/gpt2_large_uc_sft" --output_dir="saved/gpt2_large_uc_ufb_reward" --dataset_names="uc,ufb"
# accelerate launch ppo_training.py --model_name_or_path="saved/gpt2_large_uc_sft" \
#                                 --reward_model_name_or_path="saved/gpt2_large_ufb_reward" \
#                                 --output_dir="saved/gpt2_large_uc_ufb_ppo" \
#                                 --dataset_names="ufb"

