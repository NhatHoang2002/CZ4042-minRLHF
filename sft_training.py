from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig
from prepare_data import get_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, set_seed,
                          HfArgumentParser, TrainingArguments)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    # Basic training parameters
    model_name_or_path: Optional[str] = field(metadata={"help": "the location of the SFT model name or path (e.g., 'saved/model_sft/')"})
    output_dir: Optional[str] = field(metadata={"help": "the output directory (e.g., 'saved/model_dpo/')"})
    num_train_epochs: Optional[float] = field(default=2.0, metadata={"help": "number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps, set to a positive number will override num_train_epochs"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=32, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})

    # Logging and saving parameters
    logging_steps: Optional[int] = field(default=8, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=8, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=250, metadata={"help": "the evaluation frequency"})

    # Optimization parameters
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    # LoRA config
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora rank parameter"})

    # Data parameters
    dataset_names: Optional[str] = field(default="uc", metadata={"help": "the dataset names, e.g. 'uc,hh,shp'"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group the dataset by sequence length during training"})

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

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    return example["prompt"] + example["chosen"]


def create_datasets(tokenizer, args):
    # I really dont know what are the benefits of ConstantLengthDataset

    train_data = get_dataset(args.dataset_names, "train")
    valid_data = get_dataset(args.dataset_names, "test")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.max_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.max_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


if __name__ == "__main__":
    set_seed(42)
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args.dataset_names = script_args.dataset_names.split(',')
    script_args.run_name = script_args.output_dir.split('/')[-1]

    # 0. check arguments conflict
    if script_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if script_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    # 1. load a pretrained model
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        fan_in_fan_out=True,
        target_modules=["c_attn", "c_proj", "c_fc", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        token=True
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 2. Load training and evaluation datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, script_args)
    print("Train dataset length:", len(train_dataset))
    print("Eval dataset length:", len(eval_dataset))

    # 3. initialize training arguments:
    training_args = TrainingArguments(
        # Basic training parameters
        output_dir=script_args.output_dir,
        run_name=script_args.run_name,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,

        # Evaluation and logging parameters
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        logging_steps=script_args.logging_steps,
        logging_first_step=True,
        save_steps=script_args.save_steps,
        save_total_limit=3,

        # Optimization parameters
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,

        # Other parameters
        bf16=True,
        group_by_length=script_args.group_by_length,
        gradient_checkpointing=script_args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=script_args.report_to,
    )

    # 4. initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=script_args.packing,
        max_seq_length=script_args.max_length,
        peft_config=peft_config
    )
    print_trainable_parameters(trainer.model)

    # 5. train and save
    trainer.train(script_args.resume_from_checkpoint)
    trainer.save_model(script_args.output_dir)
