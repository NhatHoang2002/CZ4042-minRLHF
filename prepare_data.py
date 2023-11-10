import multiprocessing
import datasets
import pandas as pd


def get_shp(split: str, cache_dir: str = None) -> datasets.Dataset:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = []
    for row in dataset:
        scores = [row['score_A'], row['score_B']]
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue
        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        prompt = '\n### Human: ' + row['history'] + '\n### Assistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']] if row['labels'] == 1 else \
                    [' ' + row['human_ref_B'], ' ' + row['human_ref_A']]
        data.append({
            'prompt': prompt,
            'chosen': responses[0],
            'rejected': responses[1],
        })

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data))
    return dataset

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def get_hh(split: str, cache_dir: str = None) -> datasets.Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.

       Prompts should be structured as follows:
         \n### Human: <prompt>\n### Assistant:
       Multiple turns are allowed, but the prompt should always start with \n### Human: and end with \n### Assistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(samples):
        prompts, chosen_responses, rejected_responses = [], [], []
        for chosen, reject in zip(samples['chosen'], samples['rejected']):
            prompt = extract_anthropic_prompt(chosen)
            chosen_responses.append(chosen[len(prompt):])
            rejected_responses.append(reject[len(prompt):])

            prompt = prompt.replace('\n\nHuman:', '\n### Human:')
            prompt = prompt.replace('\n\nAssistant:', '\n### Assistant:')
            prompts.append(prompt)

        return {
            'prompt': prompts,
            'chosen': chosen_responses,
            'rejected': rejected_responses,
        }

    dataset = dataset.map(split_prompt_and_responses, batched=True, num_proc=multiprocessing.cpu_count())
    return dataset

def extract_ultra_data(message_list):
    prompt = ""
    for message in message_list[:-1]:
        if message['role'] == 'user':
            prompt += '\n### Human: ' + message['content']
        elif message['role'] == 'assistant':
            prompt += '\n### Assistant: ' + message['content']
        else:
            # should be no other roles
            raise ValueError(f"Unexpected role '{message['role']}'")
    prompt += '\n### Assistant:'
    response = ' ' + message_list[-1]['content']
    return prompt, response

def get_uc(split: str, cache_dir: str = None) -> datasets.Dataset:
    """Load the UltraChat dataset from Huggingface and convert it to the necessary format.

       Prompts should be structured as follows:
         \n### Human: <prompt>\n### Assistant:
       Multiple turns are allowed, but the prompt should always start with \n### Human: and end with \n### Assistant:.
    """
    print(f'Loading UC dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/ultrachat_200k', split=f"{split}_sft", cache_dir=cache_dir)
    print('done')

    data = []
    for row in dataset:
        if row['messages'][-1]['role'] != 'assistant':
            continue
        prompt, response = extract_ultra_data(row['messages'])
        data.append({
            'prompt': prompt,
            'chosen': response,
            'rejected': response
        })
    
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data))
    return dataset

def get_ufb(split: str, cache_dir: str = None) -> datasets.Dataset:
    """Load the UltraFeedback dataset from Huggingface and convert it to the necessary format.

       Prompts should be structured as follows:
         \n### Human: <prompt>\n### Assistant:
       Multiple turns are allowed, but the prompt should always start with \n### Human: and end with \n### Assistant:.
    """
    print(f'Loading UFB dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized', split=f"{split}_prefs", cache_dir=cache_dir)
    print('done')

    data = []
    for row in dataset:
        if row['messages'][-1]['role'] != 'assistant':
            continue
        prompt, chosen = extract_ultra_data(row['chosen'])
        prompt, rejected = extract_ultra_data(row['rejected'])
        data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data))
    return dataset

def get_dataset(names: str, split: str, sft_mode: bool = False, cache_dir: str = None) -> datasets.Dataset:
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    dataset = []
    for name in names:
        if name == 'shp':
            data = get_shp(split, cache_dir=cache_dir)
        elif name == 'hh':
            data = get_hh(split, cache_dir=cache_dir)
        elif name == 'uc':
            data = get_uc(split, cache_dir=cache_dir)
        elif name == 'ufb':
            data = get_ufb(split, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown dataset '{name}'")

        assert set(list(data[0].keys())) == {'prompt', 'chosen', 'rejected'}, \
            f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
        dataset.append(data)

    dataset = datasets.concatenate_datasets(dataset)
    return dataset
