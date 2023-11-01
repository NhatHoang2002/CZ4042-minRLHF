import multiprocessing
import datasets
from collections import defaultdict
import pandas as pd
import tqdm
from bs4 import BeautifulSoup, NavigableString


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text

def get_se(split, sft_mode=False, cache_dir=None):
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = []
    for row in tqdm.tqdm(dataset, desc='Processing SE'):
        prompt = '\n### Human: ' + row['question'] + '\n### Assistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE'):
        prompt = '\n### Human: ' + row['question'] + '\n### Assistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

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
    prompt_and_response = prompt_and_response.replace('\n\nHuman:', '\n### Human:')
    prompt_and_response = prompt_and_response.replace('\n\nAssistant:', '\n### Assistant:')

    search_term = '\n### Assistant:'
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
            prompts.append(prompt)
            chosen_responses.append(chosen[len(prompt):])
            rejected_responses.append(reject[len(prompt):])

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
        # elif name == 'se':
        #     data = get_se(split, sft_mode=sft_mode, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown dataset '{name}'")

        assert set(list(data[0].keys())) == {'prompt', 'chosen', 'rejected'}, \
            f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
        dataset.append(data)

    dataset = datasets.concatenate_datasets(dataset)
    return dataset
