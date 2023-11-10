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
    reward_model_name_or_path: Optional[str] = field(metadata={"help": "the reward model name or path"})
    output_dir: Optional[str] = field(metadata={"help": "the output directory (e.g., 'saved/model_dpo/')"})
    num_train_epochs: Optional[float] = field(default=5.0, metadata={"help": "number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps, set to a positive number will override num_train_epochs"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=16, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=64, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})

    # Logging and saving parameters
    logging_steps: Optional[int] = field(default=4, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=16, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})

    # Optimization parameters
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "optimizer learning rate"})
    target_kl: Optional[float] = field(default=6.0, metadata={"help": "kl target for early stopping"})
    kl_penalty: Optional[str] = field(default="kl", metadata={"help": "kl penalty type, can be 'kl' or 'kl_reverse'"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "whether to use score scaling"})
    use_score_norm: Optional[bool] = field(default=False, metadata={"help": "whether to use score normalization"})
    score_clip: Optional[float] = field(default=None, metadata={"help": "score clipping"})

    # LoRA config
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora rank parameter"})

    # Data parameters
    dataset_names: Optional[str] = field(default="hh", metadata={"help": "the dataset names, e.g. 'hh,shp'"})

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

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

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
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        peft_config=peft_config,
        trust_remote_code=True,
        token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    # 2. Load training and evaluation datasets
    train_dataset = get_dataset(script_args.dataset_names, "train")
    eval_dataset = get_dataset(script_args.dataset_names, "test")

    train_dataset = train_dataset.map(
        lambda batch: preprocess_function(tokenizer, batch, script_args.max_prompt_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=int(multiprocessing.cpu_count())
    )
    eval_dataset = eval_dataset.map(
        lambda batch: preprocess_function(tokenizer, batch, script_args.max_prom),
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=int(multiprocessing.cpu_count())
    )

    train_dataset.set_format(type='torch')
    eval_dataset.set_format(type='torch')

    print("Train dataset length:", len(train_dataset))
    print("Eval dataset length:", len(eval_dataset))

    # 3. initialize PPO config
    ppo_config = PPOConfig(
        model_name=script_args.output_dir.split('/')[-1],
        query_dataset=eval_dataset,
        reward_model=script_args.reward_model_name_or_path,
        learning_rate=script_args.learning_rate,
        log_with=None,
        mini_batch_size=script_args.per_device_train_batch_size,
        batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        early_stopping=False,
        target_kl=script_args.target_kl,
        kl_penalty=script_args.kl_penalty,
        use_score_scaling=script_args.use_score_scaling,
        use_score_norm=script_args.use_score_norm,
        score_clip=script_args.score_clip,
        seed=42
    )

    # 4. initialize the PPO trainer
    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        tokenizer,
        dataset=train_dataset,
        data_collator=collator
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin

    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            pipe = pipeline("text-classification", model=script_args.reward_model_name_or_path, device=device)
    else:
        pipe = pipeline("text-classification", model=script_args.reward_model_name_or_path, device=device)
    
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    if pipe.model.config.pad_token_id is None:
        pipe.model.config.pad_token_id = tokenizer.pad_token_id

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length": script_args.max_length,
    }

    # 5. train the model
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    
    # 6. save the model
    ppo_trainer.save_model()