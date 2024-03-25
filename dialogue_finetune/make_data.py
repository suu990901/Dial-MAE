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
                    default="bert-base-uncased",
                    required=False)
parser.add_argument('--save_to',
                    required=False,
                    default="data/ubuntu")
parser.add_argument('--data_path',
                    required=False,
                    help="Path to txt data file. One line per article format.",
                    default="data/ubuntu")
                  

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
    # concat_ids = [101] + concat_ids
    concat_ids.pop()
    return concat_ids


def encode_line_eval(line):
    label, sentences = line
    sentences = _base_encode_one_line(sentences)
    proc_lines = []
    # get whole context
    context,response = sentences[0:-1], sentences[-1]
    context = concat_with_sep(context)

    proc_lines.append({"context":context,"response":response,"label":label})
    return proc_lines

def encode_line_train(line):
    _, sentences = line
    sentences = _base_encode_one_line(sentences)
    proc_lines = []
    for i in range(1,len(sentences)):
        context = sentences[:i]
        response = sentences[i]
        context = concat_with_sep(context)
        if len(context)<=3 or len(response)==0:
            continue
        proc_line = {"context":context,"response":response,"label":-1}
        proc_lines.append(proc_line)
    return proc_lines


def main(data_path,save_path,mode="train",args=None):
    if "ubuntu" in args.data_path :
        lang="en"
    else:
        lang="zh"
    dataset = read_text_data_utterances(data_path,lang,mode=mode)
    
    if mode=="train":
        encode_line_mode = encode_line_train
    elif mode=="val" or mode=="test":
        encode_line_mode = encode_line_eval

    with open(save_path, 'w') as f:
        with Pool() as p:
            all_tokenized = p.imap_unordered(
                    encode_line_mode,
                    tqdm(dataset),
                    chunksize=2000,
                )
            for blocks in all_tokenized:
                for block in blocks:
                    f.write(json.dumps(block) + '\n')
、

if __name__=="__main__":

    train_path = args.data_path + os.sep + "train.txt"
    train_save = args.save_to + os.sep + "train"
    if os.path.exists(train_save)==False:
        os.makedirs(train_save)
    train_save = train_save + os.sep + "data.json"
    main(train_path,train_save,"train",args)

    val_path = args.data_path + os.sep + "val.txt"
    val_save = args.save_to + os.sep + "val"
    if os.path.exists(val_save)==False:
        os.makedirs(val_save)
    val_save = val_save + os.sep + "data.json"
    main(val_path,val_save,"val",args)

    test_path = args.data_path + os.sep + "test.txt"
    test_save = args.save_to + os.sep + "test"
    if os.path.exists(test_save)==False:
        os.makedirs(test_save)
    test_save = test_save + os.sep + "data.json"
    main(test_path,test_save,"test",args)
