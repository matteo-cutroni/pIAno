import json
import os
from miditok import REMI, TokenizerConfig
from collections import Counter
from tqdm import tqdm
from functools import partial
import torch


def get_top_composers(metadata_path, top_n):

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    counter = Counter()
    for file_id, content in data.items():
        meta = content.get('metadata', {})

        composer = meta.get('composer')

        if composer:
            counter[composer] += 1

    top_composers = [name for name, count in counter.most_common(top_n)]

    return top_composers

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(CURRENT_DIR, 'dataset', 'metadata.json')
DATA_PATH = os.path.join(CURRENT_DIR, "dataset", "data")
TOKENIZER_PATH = os.path.join(CURRENT_DIR, "models", "tokenizer")

# tags with which can be used to control the ai
METADATA_KEYS_TO_USE = ["composer", "genre", "period", "form", "difficulty"]

TOP_COMPOSERS = get_top_composers(METADATA_PATH, top_n=50)


def load_metadata(metadata_path):
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    # {'file_id' : ['Composer_Xxxxx', 'Genre_Xxxxx', etc...]}
    metadata_dict = {}

    for file_id, content in data.items():
        meta = content.get('metadata', {})

        meta_file_tokens = []

        for key in METADATA_KEYS_TO_USE:
            val = meta.get(key)
            if key == "composer":
                if val in TOP_COMPOSERS:
                    meta_file_tokens.append(f"Composer_{val.capitalize()}")
                else:
                    meta_file_tokens.append(f"Composer_Unknown")
            else:
                if val:
                    meta_file_tokens.append(f"{key.capitalize()}_{val.capitalize()}")
            
        metadata_dict[file_id] = meta_file_tokens

    return metadata_dict


def extract_special_tokens(metadata_dict):

    special_tokens = set()

    for meta_file_tokens in metadata_dict.values():
        special_tokens.update(meta_file_tokens)
    
    # sort so that token has always same id
    return sorted(list(special_tokens))


def setup_tokenizer(metadata_dict):

    special_tokens = extract_special_tokens(metadata_dict)

    config = TokenizerConfig(
        num_velocities=32, # default
        use_programs=False, # False because we only have one instrument (piano)
        use_tempos=True,
        use_sustain_pedals=True,
        use_time_signatures=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"] + special_tokens
    )
    tokenizer = REMI(config)
    tokenizer.save_pretrained(TOKENIZER_PATH)


def tokenize_dataset(data_path, metadata_dict, tokenizer):
    tokenized_dataset = []

    # for tqdm pbar
    file_count = sum(len(files) for _,_,files in os.walk(data_path))

    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(data_path):
            for filename in files:
                pbar.update()

                # filename format is "<file_id>_<segment_number>"
                file_id = filename.split('_')[0]

                # some files have no metadata, we exclude them
                if file_id not in metadata_dict:
                    continue

                meta_file_tokens = metadata_dict[file_id]
                meta_file_tokens_ids = [tokenizer[token] for token in meta_file_tokens]

                file_path = os.path.join(root, filename)

                tokenized_midi = tokenizer(file_path)

                full_tokenized_file = [tokenizer["BOS_None"]] + meta_file_tokens_ids + tokenized_midi + [tokenizer["EOS_None"]]

                tokenized_dataset.append(full_tokenized_file)

    return tokenized_dataset




if __name__ == "__main__":

    print("start preprocessing...")

    metadata_dict = load_metadata(METADATA_PATH)

    setup_tokenizer(metadata_dict)
    print("created tokenizer")

    tokenizer = REMI(params=os.path.join(TOKENIZER_PATH, "tokenizer.json"))
    

    tokenized_dataset = tokenize_dataset(DATA_PATH, metadata_dict, tokenizer)
    
    output_path = os.path.join(CURRENT_DIR, "dataset", "dataste.pt")
    torch.save(tokenized_dataset, output_path)
    print(f"saved tokenized dataset in: {output_path}")