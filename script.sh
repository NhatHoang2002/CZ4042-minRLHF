#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,3

accelerate launch --config_file="accelerator_config_sft.yaml" sft_training.py --model_name_or_path="gpt2-large" --output_dir="saved/gpt2_large_uc_sft" --dataset_names="uc"
accelerate launch --config_file="accelerator_config_dpo.yaml" dpo_training.py --model_name_or_path="saved/gpt2_large_uc_sft" --output_dir="saved/gpt2_large_ufb_dpo" --dataset_names="ufb"
python merge_peft_adapter.py --base_model_name="gpt2-large" --adapter_model_name="saved/gpt2_large_ufb_dpo" --output_name="saved/gpt2_large_uc_ufb_merged"


# accelerate launch reward_modeling.py --model_name_or_path="saved/gpt2_large_uc_ufb_hh_shp_sft" --output_dir="saved/gpt2_large_uc_ufb_hh_shp_reward" --dataset_names="uc,ufb,hh,shp" --max_steps 5000
# accelerate launch ppo_training.py --model_name_or_path="saved/gpt2_large_uc_ufb_hh_shp_sft" \
#                                 --reward_model_name_or_path="saved/gpt2_large_uc_ufb_hh_shp_reward" \
#                                 --output_dir="saved/gpt2_large_uc_ufb_hh_shp_ppo" \
#                                 --dataset_names="uc,ufb,hh,shp" --max_steps 5000

