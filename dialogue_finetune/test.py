import argparse
from datasets import load_dataset
from data import dialCollator
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader,Dataset
from modeling import dialForPretraining
import sys
from utils import create_optimizer,calculate_candidates_ranking,logits_recall_at_k,logits_mrr
from transformers.optimization import get_scheduler
from transformers.trainer_utils import SchedulerType
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import save_checkpoint
import utils
import math
from tqdm.auto import tqdm
import torch
from torch import nn
import numpy as np
import logging
import os
import time
from transformers import set_seed
from torch.utils.tensorboard import SummaryWriter
from main import eval_data
sys.path.append("./")

# def parser_args():
#     parser = argparse.ArgumentParser(description='train parameters')
#     parser.add_argument('--test_path', default='data/ubuntu/test/data.json', type=str)
#     parser.add_argument('--dataset', default='ubuntu', type=str)
#     parser.add_argument('--model', default="bert-base-uncased", type=str)
#     # parser.add_argument('--model', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/output/ubuntu/model/step250000_en0.3_de0.75_lr3e-4_nhead1_mul", type=str)
#     parser.add_argument('--ckpt_path', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/model/ubuntu_epoch5_lr5e-5_b64/step150000_en0.3_de0.75_lr3e-4_nhead1_mul/best/opt.pth", type=str)
#     # parser.add_argument('--model', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/output/ubuntu/model/step150000_en0.15_de0.15_lr3e-4_disable_decoder", type=str)
#     # parser.add_argument('--ckpt_path', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/model/ubuntu_epoch5_lr5e-5_b64_cuda_env_new/step150000_en0.15_de0.15_lr3e-4_disable_decoder/final/opt.pth", type=str)
#     parser.add_argument('--context_seq_length', default=256,type=int)
#     parser.add_argument('--response_seq_length', default=64,type=int)
#     parser.add_argument('--batch_size',default=1000,type=int)
#     parser.add_argument('--k_list',nargs = '+',type=int,default=[1,2,5],help="R@k,recalled top k")
#     parser.add_argument('--seed', default=42, type=int)
#     return parser.parse_args()

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--test_path', default='data/douban/test/data.json', type=str)
    parser.add_argument('--dataset', default='douban', type=str)
    parser.add_argument('--model', default="bert-base-chinese", type=str)
    # parser.add_argument('--model', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/output/douban/model/step150000_en0.15_disable_decoder_lr1e-4", type=str)
    parser.add_argument('--ckpt_path', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/save_model/en0.3_de0.45/best/opt.pth", type=str)
    parser.add_argument('--context_seq_length', default=256,type=int)
    parser.add_argument('--response_seq_length', default=64,type=int)
    parser.add_argument('--batch_size',default=180,type=int)
    parser.add_argument('--k_list',nargs = '+',type=int,default=[1,2,5],help="R@k,recalled top k")
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

# def parser_args():
#     parser = argparse.ArgumentParser(description='train parameters')
#     parser.add_argument('--test_path', default='data/rrs/test/data.json', type=str)
#     parser.add_argument('--dataset', default='rrs', type=str)
#     parser.add_argument('--model', default="bert-base-chinese", type=str)
#     parser.add_argument('--ckpt_path', default="model/chinese_mix_rrs/en0.3_de0.45/best/opt.pth", type=str)
#     parser.add_argument('--context_seq_length', default=256,type=int)
#     parser.add_argument('--response_seq_length', default=64,type=int)
#     parser.add_argument('--batch_size',default=180,type=int)
#     parser.add_argument('--k_list',nargs = '+',type=int,default=[1,2,5],help="R@k,recalled top k")
#     parser.add_argument('--seed', default=42, type=int)
#     return parser.parse_args()

# def parser_args():
#     parser = argparse.ArgumentParser(description='train parameters')
#     parser.add_argument('--test_path', default='data/ecommerce/test/data.json', type=str)
#     parser.add_argument('--dataset', default='ecommerce', type=str)
#     parser.add_argument('--model', default="bert-base-chinese")
#     # parser.add_argument('--model', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/output/ecommerce/model/step150000_en0.15_disable_decoder_lr1e-4", type=str)
#     parser.add_argument('--ckpt_path', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial_v3/dialo_finetune/model/ecommerce_eco_epoch2_lr1e-4_b128/en0.3_de0.45_v0/best/opt.pth", type=str)
#     parser.add_argument('--context_seq_length', default=256,type=int)
#     parser.add_argument('--response_seq_length', default=64,type=int)
#     parser.add_argument('--batch_size',default=1000,type=int)
#     parser.add_argument('--k_list',nargs = '+',type=int,default=[1,2,5],help="R@k,recalled top k")
#     parser.add_argument('--seed', default=42, type=int)
#     return parser.parse_args()

def test(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dialForPretraining.from_pretrained(args)
    model = utils.load_ckpt_test(model, ckpt_path=args.ckpt_path)
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    test_set = load_dataset(
        'json',
        data_files=args.test_path,
        block_size=2**25
    )['train']

    collator = dialCollator(context_seq_length=args.context_seq_length,response_seq_length=args.response_seq_length,tokenizer=tokenizer)
    test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            # shuffle=True,
            collate_fn=collator,
            drop_last=True,
            num_workers=10,
            pin_memory=True,
        )
    metrics=eval_data(model,test_loader,args,mode="test")
    print(metrics)
if __name__ == "__main__":
    args = parser_args()
    test(args)