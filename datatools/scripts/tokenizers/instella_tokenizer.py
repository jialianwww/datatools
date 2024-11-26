from __future__ import annotations

from typing import List, Optional

from tokenizers import Tokenizer as BaseTokenizer

__all__ = ["Tokenizer"]


class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around a HuggingFace :class:`tokenizers.Tokenizer`.

    :param base_tokenizer: The :class:`tokenizers.Tokenizer` to use.
    :param eos_token_id: The token ID corresponding to the "end-of-sentence" token.
    :param truncate_to: Truncate when tokenizing to this number of token IDs.
    :param truncate_direction: The direction to truncate in. "right" means truncate the tokens
        on the right. "left" means truncate the tokens on the left. If ``truncate_to`` is null,
        this setting has no effect.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        truncate_to: Optional[int] = None,
        truncate_direction: str = "right",
    ):
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.no_truncation()
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.truncate_to = truncate_to
        self.truncate_direction = truncate_direction

    @property
    def vocab_size(self) -> int:
        return self.base_tokenizer.get_vocab_size()

    @property
    def eos_token(self) -> str:
        return self.decode([self.eos_token_id], skip_special_tokens=False)

    @property
    def pad_token(self) -> str:
        return self.decode([self.pad_token_id], skip_special_tokens=False)

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_file(cls, filename, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_file(str(filename))
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if not input_ids or input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1

    def _truncate(
        self, input_ids: List[int], truncate_to: Optional[int], direction: str
    ) -> list[int]:
        if truncate_to is None or len(input_ids) <= truncate_to:
            return input_ids
        elif direction == "left":
            return input_ids[len(input_ids) - truncate_to :]
        else:
            return input_ids[: -(len(input_ids) - truncate_to)]

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.
        """
        return self.encode_batch([input], add_special_tokens=add_special_tokens)[0]

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = self._truncate(encoding.ids, truncate_to, self.truncate_direction)
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)
            all_input_ids.append(input_ids)

        return all_input_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs to a string.
        """
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
