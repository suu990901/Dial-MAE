#!/usr/bin/python
# -*- encoding: utf-8 -*-
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import os
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask

from transformers.utils import logging
logger = logging.get_logger(__name__)

@dataclass
class CotMAECollator(DataCollatorForWholeWordMask):
    context_seq_length: int = 512
    response_seq_length: int = 512
    encoder_mask_ratio: float = 0.15
    decoder_mask_ratio: float = 0.15

    # def __post_init__(self):
    #     super().__post_init__()
    #     self.rng = random.Random()
    #     self.rng.seed(42)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.15):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _truncate(self, example: List[int],mode="context"):
        if mode == "context":
            tgt_len = self.context_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        else:
            tgt_len = self.response_seq_length - self.tokenizer.num_special_tokens_to_add(False)
            
        if len(example) <= tgt_len:
            return example
   
        if mode == "context":
            truncated = example[-tgt_len:]
        elif mode == "response":
            truncated = example[:tgt_len]
        else:
            raise ValueError("truncate mode error")

        return truncated

    def _pad(self, seq, mode, val=0):
        tgt_len = self.context_seq_length if mode=="context" else self.response_seq_length
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]
    
    def encode_batch_examples(self, examples: List[Dict[str, List[int]]], mode, mlm_prob: float=0.15):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:
            e_text = e['text']
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_text]
            mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)
            mlm_mask = self._pad([0] + mlm_mask,mode)
            mlm_masks.append(mlm_mask)

            max_length = self.context_seq_length if mode=="context" else self.response_seq_length
            encoded = self.tokenizer.encode_plus(
                e_text,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

        inputs, labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )
        attention_mask = torch.tensor(masks)

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
            # "input_ids_unmasked": torch.tensor(encoded_examples, dtype=torch.long),
        }

        return batch

    
    def __call__(self, examples):
        context = []
        response = []
        for text_dict in examples:
            context.append({'text': self._truncate(text_dict['context'],"context")})
            response.append({'text': self._truncate(text_dict['response'],"response")})
        context_batch = self.encode_batch_examples(examples=context, mode="context", mlm_prob=self.encoder_mask_ratio)
        response_batch = self.encode_batch_examples(examples=response, mode="response",mlm_prob=self.decoder_mask_ratio)
        context_batch['decoder_input_ids'] = response_batch['input_ids']
        context_batch['decoder_labels'] = response_batch['labels']
        context_batch['decoder_attention_mask'] = response_batch['attention_mask']
        return context_batch

