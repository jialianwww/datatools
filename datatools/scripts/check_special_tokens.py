from streaming import LocalDataset
from datatools.scripts.tokenizers.llama3_tokenizer import Tokenizer
import os

def check(dataset_path):
    dataset = LocalDataset(dataset_path)

    tokenizer = Tokenizer("scripts/tokenizers/llama3_tokenizer.model")

    special_tokens = tokenizer.special_tokens
    special_tokens_to_check = []
    for token, token_id in special_tokens.items():
        if 'reserve' not in token:
            special_tokens_to_check.append(token)

    special_tokens_found = []
    for sample in dataset:
        text = tokenizer.decode(sample["input_ids"])
        for special_token in special_tokens_to_check:
            if special_token in text and special_token not in special_tokens_found:
                special_tokens_found.append(special_token)
    
    print('There are following special token text appearing in the text of the dataset:')
    print(special_tokens_found)


if __name__ == "__main__":
    data_folder = "/dockerx/media/4TB/users/jialiawu/datasets/prolong/prolong_raw"
    dataset = "textbooks"
    # dataset = "tuluv2"
    check(os.path.join(data_folder, dataset))