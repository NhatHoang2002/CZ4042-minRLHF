import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig
from prepare_data import get_dataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer


@dataclass
class ScriptArguments:
    # Basic training parameters
    model_name_or_path: Optional[str] = field(metadata={"help": "the location of the SFT model name or path (e.g., 'saved/model_sft/')"})
    output_dir: Optional[str] = field(metadata={"help": "the output directory (e.g., 'saved/model_dpo/')"})
    num_train_epochs: Optional[float] = field(default=3.0, metadata={"help": "number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps, set to a positive number will override num_train_epochs"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=16, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})

    # Logging and saving parameters
    logging_steps: Optional[int] = field(default=4, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=16, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency"})

    # Optimization parameters
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    # LoRA config
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # Data parameters
    dataset_names: Optional[str] = field(default="hh", metadata={"help": "the dataset names, e.g. 'hh', 'shp'"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # Sequence length parameters
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})

    # Instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    # Debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args.dataset_names = script_args.dataset_names.split(',')

    # 0. check arguments conflict
    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if script_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    # 1. load a pretrained model
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=True
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load training and evaluation datasets
    train_dataset = get_dataset(script_args.dataset_names, "train")
    eval_dataset = get_dataset(script_args.dataset_names, "test")

    # 3. initialize training arguments:
    training_args = TrainingArguments(
        # Basic training parameters
        output_dir=script_args.output_dir,
        run_name=script_args.output_dir.split('/')[-1],
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,

        # Evaluation and logging parameters
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,

        # Optimization parameters
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,

        # Other parameters
        bf16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=script_args.report_to,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train(script_args.resume_from_checkpoint)
    dpo_trainer.save_model(script_args.output_dir)
