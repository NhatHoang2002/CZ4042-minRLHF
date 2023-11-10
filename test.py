import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          set_seed)
from argparse import ArgumentParser
from prepare_data import get_dataset


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = 'left'

    tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset([args.eval_data], 'test')
    dataset = dataset.filter(lambda x: len(tokenizer(x['prompt'])['input_ids']) <= 512)
    dataset = dataset.map(lambda x: tokenizer(x['prompt'], padding='max_length', max_length=512, truncation=True), batched=True)

    if args.sanity_check:
        dataset = dataset.select(range(10))

    dataloader = DataLoader(dataset, batch_size=8, 
                            collate_fn=lambda x: (torch.as_tensor([x_['input_ids'] for x_ in x]),
                                                  [x_['prompt'] for x_ in x],
                                                  [x_['chosen'] for x_ in x],
                                                  [x_['rejected'] for x_ in x]),
                            shuffle=False,
                            num_workers=4,
                            drop_last=False
    )

    prompts = []
    preds = []
    chosen = []
    rejected = []

    for (b_input_ids, b_prompt, b_chosen, b_rejected) in tqdm(dataloader, total=len(dataloader)):
        b_input_ids = b_input_ids.to(device)
        outputs = model.generate(
            b_input_ids,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_length=1024,
            top_p=0.95,
            top_k=50,
            num_beams=4,
            num_return_sequences=1,        
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        prompts.extend(b_prompt)
        preds.extend(outputs)
        chosen.extend(b_chosen)
        rejected.extend(b_rejected)

    df = pd.DataFrame({
        'prompt': prompts,
        'pred': preds,
        'chosen': chosen,
        'rejected': rejected
    })

    model_name = args.model_name_or_path.split('/')[-1]
    df.to_csv(f'output_{args.eval_data}_{model_name}.csv', index=False, encoding='utf_8_sig')


if __name__ == "__main__":
    set_seed(42)
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2-large")
    parser.add_argument("--eval_data", type=str, default="hh")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=1 python test.py --model_name_or_path saved/gpt2_large_uc_ufb_merged --eval_data hh
# CUDA_VISIBLE_DEVICES=2 python test.py --model_name_or_path saved/gpt2_medium_uc_hh_ufb_merged --eval_data uc