import json
import os
from miditok import REMI, TokenizerConfig
from miditok.constants import CHORD_MAPS
from collections import Counter

def get_top_composers(json_path, top_n):

    with open(json_path, 'r') as f:
        data = json.load(f)

    counter = Counter()
    for file_id, content in data.items():
        meta = content.get('metadata', {})

        composer = meta.get('composer')

        if composer:
            counter[composer] += 1

    top_composers = [name for name, count in counter.most_common(top_n)]

    return top_composers

def load_metadata(json_path):
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # {'file_id' : ['Composer_Xxxxx', 'Genre_Xxxxx', etc...]}
    metadata_dict = {}

    for file_id, content in data.items():
        meta = content.get('metadata', {})

        file_tokens = []

        for key in METADATA_KEYS_TO_USE:
            val = meta.get(key)
            if key == "composer":
                if val in TOP_COMPOSERS:
                    file_tokens.append(f"Composer_{val.capitalize()}")
                else:
                    file_tokens.append(f"Composer_Unknown")
            else:
                if val:
                    file_tokens.append(f"{key.capitalize()}_{val.capitalize()}")
            
        metadata_dict[file_id] = file_tokens

    return metadata_dict



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

METADATA_PATH = os.path.join(CURRENT_DIR, 'dataset', 'metadata.json')

# tags with which can be used to control the ai
METADATA_KEYS_TO_USE = ["composer", "genre", "music_period", "form", "difficulty"]

TOP_COMPOSERS = get_top_composers(METADATA_PATH, top_n=50)


if __name__ == "__main__":

    print(f"these are the top 5 composers in the dataset: {TOP_COMPOSERS[:5]}")


