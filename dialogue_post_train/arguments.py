#!/usr/bin/python
# -*- encoding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, Union
import os
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments control input data path, mask behaviors
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    context_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total context sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    response_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total response sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    min_seq_length: int = field(default=16)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    encoder_mask_ratio: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for encoder"}
    )
    decoder_mask_ratio: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for decoder"}
    )

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]


@dataclass
class ModelArguments:
    """
    Arguments control model config, decoder head config
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='bert',
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    n_head_layers: int = field(default=2)
    use_decoder_head: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to use decoder head of transformer based MAE, please set to True"}
    )

    """
    head mlm
    """
    enable_head_mlm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add decode-head layer mlm loss"}
    )
    head_mlm_coef: Optional[float] = field(default=1.0)


@dataclass
class CotMAEPreTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)
