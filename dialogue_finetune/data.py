import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask

@dataclass
class dialCollator():

    def __init__(self,context_seq_length,response_seq_length,tokenizer):
        self.context_seq_length = context_seq_length
        self.response_seq_length = response_seq_length
        self.tokenizer = tokenizer
 
    def _truncate(self, example: List[int],mode="context"):
        if mode=="context":
            tgt_len = self.context_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        elif mode=="response":
            tgt_len = self.response_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        else:
            raise ValueError("truncate mode error")
        if len(example) <= tgt_len:
            return example
   
        if mode == "context":
            truncated = example[-tgt_len:]
        elif mode == "response":
            truncated = example[:tgt_len]
        else:
            raise ValueError("truncate mode error")

        return truncated


    def __call__(self, examples: List[Dict[str, List[int]]]):
        context_encoded_examples,response_encoded_examples = [],[]
        context_masks,response_masks  = [],[]
        labels = []
        for e in examples:
            context = self._truncate(e['context'],mode="context")
            response = self._truncate(e['response'],mode="response")
            labels.append(e['label'])
            encoded_context = self.tokenizer.encode_plus(
                context,
                add_special_tokens=True,
                max_length=self.context_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
            encoded_response = self.tokenizer.encode_plus(
                response,
                add_special_tokens=True,
                max_length=self.response_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )

            context_masks.append(encoded_context['attention_mask'])
            context_encoded_examples.append(encoded_context['input_ids'])

            response_masks.append(encoded_response['attention_mask'])
            response_encoded_examples.append(encoded_response['input_ids'])

        batch = {
            "context_input_ids": torch.tensor(context_encoded_examples,dtype=torch.long),
            "context_attention_mask": torch.tensor(context_masks,dtype=torch.long),
            "response_input_ids": torch.tensor(response_encoded_examples,dtype=torch.long),
            "response_attention_mask": torch.tensor(response_masks,dtype=torch.long),
            "labels":torch.tensor(labels,dtype=torch.long),
        }

        return batch

