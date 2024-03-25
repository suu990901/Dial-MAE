import os
import random
import argparse
from typing import List
from math import floor
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from functools import partial
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer',
                    default="bert-base-chinese",
                    required=False)
parser.add_argument('--min_span',
                    default=1,
                    type=int,
                    required=False,
                    help="Sliding window minimum size ",
                    )
parser.add_argument('--max_span',
                    default=20,
                    type=int,
                    required=False,
                    help="sliding window maximum size ",
                    )
parser.add_argument('--num_window',
                    default=6,
                    type=int,
                    required=False,
                    help="num window ",
                    )
parser.add_argument('--save_to',
                    required=False,
                    default="data/ecommerce_simple")
parser.add_argument('--data_path',
                    required=False,
                    help="Path to txt data file. One line per article format.",
                    default="data/ecommerce_simple")
                  


args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

def _base_encode_one_line(sentences: list) -> List[List[List[int]]]:
    tokenized = [
        tokenizer(
            s,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for s in sentences
    ]
    return tokenized

def read_text_data_utterances(path, lang='zh',mode="train"):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            # line="\t".join(line.split())
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            utterances = [i for i in utterances if i!='']
            if label==0 and mode=="train":
                continue
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            dataset.append((label, utterances))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset

def concat_with_sep(current_block):

    concat_ids = []
    for i in current_block:
        concat_ids.extend(i)
        concat_ids.append(102)
    concat_ids.pop()
    return concat_ids

def encode_line(line,mode="train"):

    min_span,max_span=args.min_span,args.max_span
    label, sentences = line
    sentences = _base_encode_one_line(sentences)
    proc_lines = []
    context,response = sentences[0:-1], sentences[-1]
    context = concat_with_sep(context)
    if len(context)!=0 and len(response)!=0:
        proc_lines.append({"context":context,"response":response})

    if mode!="train":
        return proc_lines
    if len(sentences)<args.min_span:
        return proc_lines

    end_index = min(len(sentences)-min_span,len(sentences)-1)
    for start in range(end_index):
        res_span_len = len(sentences) - start -1
        max_index = min(max_span,res_span_len)
        min_index = min(min_span,res_span_len)
        size = min(max_index-min_index+1,args.num_window)
        slides = np.random.choice(range(min_index,max_index+1),size=size,replace=False)
        for slide_size in slides:
            end = start + slide_size
            if start == 0 and end == len(sentences)-1:
                continue
            context = sentences[start:end]
            response = sentences[end]
            context = concat_with_sep(context)
            if len(context)==0 or len(response)==0:
                continue
            proc_lines.append({"context":context,"response":response})

    if len(proc_lines)==0:
        return None
    return proc_lines


def main(data_path,save_path,mode="train",args=None):
    if "ubuntu" in args.data_path :
        lang="en"
    else:
        lang="zh"
    dataset = read_text_data_utterances(data_path,lang,mode=mode)
    
    encode_line_mode = partial(encode_line,mode=mode)
    with open(save_path, 'w') as f:
        with Pool() as p:
            all_tokenized = p.imap_unordered(
                        encode_line_mode,
                        tqdm(dataset),
                        chunksize=2000,
                    )
            for blocks in all_tokenized:
                if blocks is None:
                    continue
                for block in blocks:
                    f.write(json.dumps(block) + '\n')
    # debug               
    # all_lines = []
    # for line in tqdm(dataset):
    #     proc_lines = encode_line(line,mode=mode)
    #     all_lines.extend(proc_lines)

    # with open(save_path, 'w') as f:
    #     for line in all_lines:
    #         f.write(json.dumps(line) + '\n')

if __name__=="__main__":

    train_path = args.data_path + os.sep + "train.txt"
    train_save = args.save_to + os.sep + "train"
    if os.path.exists(train_save)==False:
        os.makedirs(train_save)
    train_save = train_save + os.sep + "data.json"
    main(train_path,train_save,"train",args)

