import argparse
from datasets import load_dataset
from data import dialCollator
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader,Dataset
from modeling import dialForPretraining
import sys
from utils import create_optimizer,calculate_candidates_ranking,logits_recall_at_k,logits_mrr
from utils import mean_average_precision,precision_at_one
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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
sys.path.append("./")

response_number={
    "ubuntu_val":10,
    "ubuntu_test":10,
    "douban_val":2,
    "douban_test":10,
    "ecommerce_val":2,
    "ecommerce_test":10,
}

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--train_path', default='ubuntu/train/data.json', type=str)
    parser.add_argument('--val_path', default='ubuntu/val/data.json', type=str)
    parser.add_argument('--dataset', default='ubuntu', type=str)
    parser.add_argument('--model', default="bert-base-uncased", type=str)
    parser.add_argument('--ckpt_path', default="bert-fp-mono/bert-base-uncased_cpu.pt", type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--context_seq_length', default=256,type=int)
    parser.add_argument('--response_seq_length', default=64,type=int)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--save_path', default='save_model', type=str,help="save folder")
    parser.add_argument('--eval_mode', default='test', type=str,help="save folder")
    parser.add_argument('--tensorboard_dir', default='ubuntu_log', type=str,help="save folder")
    parser.add_argument('--temp',default=0.07,type=float,help="contrastive learning temperature")
    parser.add_argument('--lr',default=5e-5,type=float)
    parser.add_argument('--weight_decay',default=0,type=float)
    parser.add_argument('--warm_up_ratio',default=0.0,type=float)
    parser.add_argument('--epochs',default=5,type=int)
    parser.add_argument('--clip_grad_norm',default=5.0,type=float,help="grad clip")
    parser.add_argument('--use_cross_cl',default=False,type=bool)
    parser.add_argument('--k_list',nargs = '+',type=int,default=[1,2,5],help="R@k,recalled top k")
    parser.add_argument('--eval_steps', default=-1, type=int,
                        help='eval steps')
    parser.add_argument('--num_eval_times', default=10, type=int,
                        help='number of verifications')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()

@torch.no_grad()
def eval_data(model,eval_loader,args,mode="test"):
    model.eval()
    cos = nn.CosineSimilarity(dim=-1)
    all_scores,all_labels = [],[]
    for batch in tqdm(eval_loader):
        context_cls_hiddens,response_cls_hiddens,labels = model(batch,sent_emb=True,eval_mode=mode)
        scores = torch.mul(context_cls_hiddens,response_cls_hiddens)
        scores = scores.sum(dim=-1)
        # scores = cos(context_cls_hiddens,response_cls_hiddens)
        all_scores.append(scores)
        all_labels.append(labels)
    
    all_scores,all_labels=torch.cat(all_scores,dim=0),torch.cat(all_labels,dim=0)
    total_mrr,total_correct=0,0
    total_examples = 0
    total_prec_at_one,total_map=0,0
    for index in range(all_scores.shape[0]):
        scores,label = all_scores[index],all_labels[index]
        if args.dataset == "ubuntu" or mode=="test":
            rank_by_pred, pos_index, stack_scores = \
                calculate_candidates_ranking(
                    np.array(scores.cpu().tolist()), 
                    np.array(label.cpu().tolist()),
                    10)
        else:
            rank_by_pred, pos_index, stack_scores = \
                calculate_candidates_ranking(
                    np.array(scores.cpu().tolist()), 
                    np.array(label.cpu().tolist()),
                    2)
        # some douban data have not true labels
        if sum(rank_by_pred[0])==0 and args.dataset == "douban":
            continue
        num_correct = logits_recall_at_k(pos_index, args.k_list)
        total_mrr += logits_mrr(pos_index)
        total_correct = np.add(total_correct, num_correct)
        total_prec_at_one += precision_at_one(rank_by_pred)
        total_map += mean_average_precision(pos_index)
        total_examples += 1
    # total_examples=all_scores.shape[0]
    avg_mrr = float(total_mrr / total_examples)
    
    if args.dataset == "douban" and mode=='test':
        p_1 = float(total_prec_at_one / total_examples)
        map = float(total_map / total_examples)
        R10_1 = round(((total_correct[0]/total_examples)*100), 2)
        R10_2 = round(((total_correct[1]/total_examples)*100), 2)
        R10_5 = round(((total_correct[2]/total_examples)*100), 2)
        return avg_mrr,R10_1,R10_2,R10_5,p_1,map

    if args.dataset == "ubuntu" or mode=='test':
        R10_1 = round(((total_correct[0]/total_examples)*100), 2)
        R10_2 = round(((total_correct[1]/total_examples)*100), 2)
        R10_5 = round(((total_correct[2]/total_examples)*100), 2)
        print(R10_1,R10_2,R10_5,total_examples)
        return avg_mrr,R10_1,R10_2,R10_5
    else:
        R2_1 = round(((total_correct[0]/total_examples)*100), 2)
        return avg_mrr,R2_1

def train(model,train_loader,dev_loader,optimizer,loss_scaler,scheduler,best_score,epoch,writer,args):
    model.train()
    if utils.get_rank()==0:
        steps_trained_progress_bar = tqdm(total=len(train_loader))
    for step,batch in enumerate(train_loader):
        global_step = args.num_training_steps_per_epoch * epoch + step
        with torch.cuda.amp.autocast():
            loss = model(batch,sent_emb=False)
            optimizer.zero_grad()
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad_norm,
                                    parameters=model.parameters())
            loss_scale_value = loss_scaler.state_dict()["scale"]

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            scheduler.step()

            if writer is not None:
                writer.add_scalar('loss', loss.item(), global_step)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss_scaler', loss_scaler.state_dict()["scale"], global_step)

            if utils.get_rank()==0:
                steps_trained_progress_bar.update(1)

        if global_step%args.eval_steps==0:
            if args.dataset == "ubuntu" or args.eval_mode=="test":
                avg_mrr,R10_1,R10_2,R10_5=eval_data(model,dev_loader,args,args.eval_mode)
                logger.info(f"rank:{utils.get_rank()},epoch:{epoch},global_step:{global_step},avg_mrr:{avg_mrr},R10_1:{R10_1},R10_2:{R10_2},R10_5:{R10_5}")
                score_now = R10_1+R10_2+R10_5
            else:
                avg_mrr,R2_1=eval_data(model,dev_loader,args,args.eval_mode)
                logger.info(f"rank:{utils.get_rank()},epoch:{epoch},global_step:{global_step},avg_mrr:{avg_mrr},R2_1:{R2_1}")
                score_now = R2_1
            model.train()
            if score_now>best_score:
                best_score = score_now
                logger.info(f"save best model ...")
                model_without_ddp = model
                if args.distributed:
                    model_without_ddp = model.module
                save_checkpoint(args,model,model_without_ddp,optimizer,loss_scaler,mode="step")
    return best_score
            
def main():
    args = parser_args()
    utils.init_distributed_mode(args)
    set_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dialForPretraining.from_pretrained(args)
    # model = utils.load_ckpt(model, ckpt_path=args.ckpt_path)
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    train_set = load_dataset(
        'json',
        data_files=args.train_path,
        block_size=2**25,
    )['train']
    dev_set = load_dataset(
        'json',
        data_files=args.val_path,
        block_size=2**25
    )['train'] \
        if args.val_path is not None else None
    collator = dialCollator(context_seq_length=args.context_seq_length,response_seq_length=args.response_seq_length,tokenizer=tokenizer)

    if args.distributed: 
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        sampler_train = torch.utils.data.DistributedSampler(
            train_set, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)

    train_loader = DataLoader(
            train_set,
            sampler=sampler_train,
            batch_size=args.batch_size,
            collate_fn=collator,
            drop_last=True,
            num_workers=10,
            pin_memory=True,
        )
    dev_loader = DataLoader(
            dev_set,
            batch_size=400,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=10,
            pin_memory=True,
        )

    num_tasks = utils.get_world_size()
    num_training_steps_per_epoch = math.ceil(len(train_set) // args.batch_size // num_tasks)
    num_training_steps = num_training_steps_per_epoch * args.epochs
    args.num_training_steps_per_epoch=num_training_steps_per_epoch
    num_warmup_steps = num_training_steps * args.warm_up_ratio
    if args.eval_steps==-1:
        args.eval_steps=num_training_steps//args.num_eval_times

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        logger.setLevel(logging.INFO if utils.get_rank() % 8==0 else logging.ERROR)
    
    optimizer = create_optimizer(args,model_without_ddp)
    scheduler = get_scheduler(
                SchedulerType.LINEAR,
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
    loss_scaler =  NativeScaler() 
    # train(model,train_loader,optimizer,loss_scaler,scheduler,args)
    # eval_ubuntu(model,dev_loader,optimizer,args)
    best_score = -1
    now = int(time.time())
    timearr = time.localtime(now)
    timearr = time.strftime("%Y_%m_%d_%H%M%S", timearr)
    # tensorboard_path = args.tensorboard_dir + os.sep + timearr
    tensorboard_path = args.tensorboard_dir 

    if utils.get_rank()==0:
        if os.path.exists(tensorboard_path)==False:
            os.makedirs(tensorboard_path)
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None
        
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        best_score=train(model,train_loader,dev_loader,optimizer,loss_scaler,scheduler,best_score,epoch,writer,args)
    logger.info(f"save final model ...")
    save_checkpoint(args,model,model_without_ddp,optimizer,loss_scaler,mode="final",tokenizer=tokenizer)
        
if __name__=="__main__":
    main()
