# RLHF with Small-scaled Models

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
pip install -r requirements.txt
```

Then run `accelerate config` to configure the distributed training environment.

## Training

First, perform SFT training on the base model using the `sft_training.py` script. Then fine-tune the model using PPO or DPO.

The following command trains the model on the UC dataset using the GPT-2 medium model in SFT setting:
```
accelerate launch sft_training.py \
        --model_name_or_path="gpt2-medium" \
        --output_dir="saved/gpt2_medium_uc_sft" \
        --dataset_names="uc"
```

The following command trains the model on the UFB dataset using the GPT-2 medium model in DPO setting. Then merge the model with the base model because LoRA was used:
```
# DPO training
accelerate launch dpo_training.py \
        --model_name_or_path="saved/gpt2_medium_uc_sft" \
        --output_dir="saved/gpt2_medium_ufb_dpo" \
        --dataset_names="ufb"

# Merge model
python merge_peft_adapter.py \
        --base_model_name="gpt2-medium" \
        --adapter_model_name="saved/gpt2_medium_ufb_dpo" \
        --output_name="saved/gpt2_medium_uc_ufb_merged"
```

The following command trains the model on the HH dataset using the GPT-2 medium model in PPO setting:
```
# Train reward model
accelerate launch reward_modeling.py \
        --model_name_or_path="saved/gpt2_large_uc_ufb_hh_shp_sft" \
        --output_dir="saved/gpt2_large_uc_ufb_hh_shp_reward" \
        --dataset_names="uc,ufb,hh,shp"

# PPO training
accelerate launch ppo_training.py --model_name_or_path="saved/gpt2_large_uc_ufb_hh_shp_sft" \
        --reward_model_name_or_path="saved/gpt2_large_uc_ufb_hh_shp_reward" \
        --output_dir="saved/gpt2_large_uc_ufb_hh_shp_ppo" \
        --dataset_names="uc,ufb,hh,shp"
```

Note that we can add as many datasets as we want to the `dataset_names` argument, separated by commas. See the file `prepare_data.py` for the available datasets.

## Acknowledgements

This small project is part of the CZ4042 Neural Networks course at NTU. The code is largely based on the HuggingFace's [Transformer Reinforcement Learning (TRL)](https://github.com/huggingface/trl) library.
