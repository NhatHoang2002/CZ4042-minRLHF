import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from prepare_data import get_dataset
from tqdm import tqdm
from transformers import (Adafactor, AutoTokenizer, HfArgumentParser, pipeline,
                          set_seed)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

tqdm.pandas()


@dataclass
class ScriptArguments:
    # Basic training parameters
    model_name_or_path: Optional[str] = field(metadata={"help": "the location of the SFT model name or path (e.g., 'saved/model_sft/')"})
    reward_model_name_or_path: Optional[str] = field(default="", metadata={"help": "the reward model name or path"})
    output_dir: Optional[str] = field(metadata={"help": "the output directory (e.g., 'saved/model_dpo/')"})
    num_train_epochs: Optional[float] = field(default=5.0, metadata={"help": "number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps, set to a positive number will override num_train_epochs"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=32, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})

    # Logging and saving parameters
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


def preprocess_function(tokenizer, examples, max_prompt_length):
    new_examples = {
        "input_ids": [],
        "query": []
    }

    encoded_prompt = tokenizer.encode(examples["prompt"], padding='max_length', max_length=max_prompt_length)
    for prompt, e_prompt in zip(examples['prompt'], encoded_prompt):
        if e_prompt > max_prompt_length:
            continue
        new_examples["input_ids"].append(e_prompt)
        new_examples["query"].append(prompt)
    
    return new_examples


if __name__ == "__main__":
    set_seed(42)
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args.dataset_names = script_args.dataset_names.split(',')

    # 0. check arguments conflict
    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if script_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    # 1. load a pretrained model
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["c_attn", "c_proj", "c_fc", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, peft_config=peft_config)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    # 2. Load training and evaluation datasets
    train_dataset = get_dataset(script_args.dataset_names, "train")
    eval_dataset = get_dataset(script_args.dataset_names, "test")

    train_dataset = train_dataset.map(
        lambda batch: preprocess_function(tokenizer, batch, script_args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=int(multiprocessing.cpu_count())
    )
    eval_dataset = eval_dataset.map(
        lambda batch: preprocess_function(tokenizer, batch, script_args.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=int(multiprocessing.cpu_count())
    )

    train_dataset.set_format(type='torch')
    eval_dataset.set_format(type='torch')

    print("Train dataset length:", len(train_dataset))
    print("Eval dataset length:", len(eval_dataset))

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # 3. initialize training arguments:

    