from functools import partial

from tqdm import tqdm

from typing import Optional

import numpy as np
from pathlib import Path

from simple_parsing import ArgumentParser, field
from dataclasses import dataclass

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions
from streaming.base.array import Array


@dataclass
class DecodeOptions:
    """Options for decoding"""

    domain_field: str = "domain"
    dataset_field: str = "dataset"

    def __post_init__(self):
        pass


def decode_fn(data: Array,
                process_id: int,
                options: DecodeOptions):

    from datatools.scripts.tokenizers.llama3_tokenizer import Tokenizer
    tokenizer = Tokenizer(str(Path(__file__).parent / "tokenizers" / "llama3_tokenizer.model"))
    
    print('The dataset has keys:')
    print(data[0].keys())

    for i in tqdm(range(len(data)), disable=process_id != 0):
        item = data[i]
        
        assert item["input_ids"].ndim == 1 and item["input_ids"].dtype == np.uint32
        input_ids = [int(x) for x in item["input_ids"]]
        text = tokenizer.decode(input_ids)

        output_item = {
            "text": text,
        }

        if options.domain_field in item:
            output_item[options.domain_field] = item[options.domain_field]
        if options.dataset_field in item:
            output_item[options.dataset_field] = item[options.dataset_field]


        yield output_item


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(DecodeOptions, dest="decode_options")
    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(dataset,
            partial(decode_fn, options=args.decode_options),
            args.output,
            args.process_options)


if __name__ == "__main__":
    main()
