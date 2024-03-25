import torch 
from torch import nn
from torch import optim as optim
import torch.distributed as dist
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
import os
from pathlib import Path
def create_optimizer(args, model):

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(parameters, **opt_args)
    return optimizer

def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")
        
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict): 
        self._scaler.load_state_dict(state_dict)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

'''https://github.com/taesunwhang/BERT-ResSel/blob/master/evaluation.py'''
def calculate_candidates_ranking(prediction, ground_truth, eval_candidates_num=10):
    total_num_split = len(ground_truth) / eval_candidates_num
    pred_split = np.split(prediction, total_num_split)
    gt_split = np.split(np.array(ground_truth), total_num_split)
    orig_rank_split = np.split(np.tile(np.arange(0, eval_candidates_num), int(total_num_split)), total_num_split)
    stack_scores = np.stack((gt_split, pred_split, orig_rank_split), axis=-1)
    
    rank_by_pred_l = []
    for i, stack_score in enumerate(stack_scores):
        rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
        rank_by_pred = np.stack(rank_by_pred, axis=-1)
        rank_by_pred_l.append(rank_by_pred[0])
    rank_by_pred = np.array(rank_by_pred_l)
    
    pos_index = []
    for sorted_score in rank_by_pred:
        curr_cand = []
        for p_i, score in enumerate(sorted_score):
            if int(score) == 1:
                curr_cand.append(p_i)
        pos_index.append(curr_cand)

    return rank_by_pred, pos_index, stack_scores

def logits_recall_at_k(pos_index, k_list=[1, 2, 5, 10]):
    # 1 dialog, 10 response candidates ground truth 1 or 0
    # prediction_score : [batch_size]
    # target : [batch_size] e.g. 1 0 0 0 0 0 0 0 0 0
    # e.g. batch : 100 -> 100/10 = 10
    num_correct = np.zeros([len(pos_index), len(k_list)])
    index_dict = dict()
    for i, p_i in enumerate(pos_index):
        index_dict[i] = p_i

    # case for douban : more than one correct answer case
    for i, p_i in enumerate(pos_index):
        if len(p_i) == 1 and p_i[0] >= 0:
            for j, k in enumerate(k_list):
                if p_i[0] + 1 <= k:
                    num_correct[i][j] += 1
        elif len(p_i) > 1:
            for j, k in enumerate(k_list):
                all_recall_at_k = []
                for cand_p_i in p_i:
                    if cand_p_i + 1 <= k:
                        all_recall_at_k.append(1)
                    else:
                        all_recall_at_k.append(0)
                num_correct[i][j] += np.mean(all_recall_at_k)
                # num_correct[i][j] += np.max(all_recall_at_k)

    return np.sum(num_correct, axis=0)

def logits_mrr(pos_index):
    mrr = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0 and p_i[0] >= 0:
            mrr.append(1 / (p_i[0] + 1))
        elif len(p_i) == 0:
            mrr.append(0)  # no answer

    return np.sum(mrr)

def mean_average_precision(pos_index):
    map = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0:
            all_precision = []
            for j, cand_p_i in enumerate(p_i):
                all_precision.append((j + 1) / (cand_p_i + 1))
            curr_map = np.mean(all_precision)
            map.append(curr_map)
        elif len(p_i) == 0:
            map.append(0)  # no answer
    return np.sum(map)

def precision_at_one(rank_by_pred):
    num_correct = [0] * rank_by_pred.shape[0]
    for i, sorted_score in enumerate(rank_by_pred):
        for p_i, score in enumerate(sorted_score):
            if p_i == 0 and int(score) == 1:
                num_correct[i] = 1
                break
    return np.sum(num_correct)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_checkpoint(args, model, model_without_ddp, optimizer, loss_scaler, mode="step",tokenizer=None):
    output_dir = Path(args.save_path)
    
    if mode == "step":
        checkpoint_path = output_dir / 'best'
        opt_path = checkpoint_path / 'opt.pth'
    else:
        checkpoint_path = output_dir / 'final'
        opt_path = checkpoint_path / 'opt.pth'

    if checkpoint_path.exists()==False and is_main_process():
        os.makedirs(checkpoint_path)

    # if is_main_process():
    #     model_without_ddp.lm.save_pretrained(checkpoint_path)

    to_save = {
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'model':model_without_ddp.state_dict()
        }
    save_on_master(to_save, opt_path)

    if is_main_process() and mode=='final':
        tokenizer.save_pretrained(output_dir / "best")
        tokenizer.save_pretrained(output_dir / "final")

def load_ckpt(model,ckpt_path):
    ckpt = torch.load(ckpt_path)
    # ckpt = ckpt.to("cpu")
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if "context_lm." in key:
            model_key_prefix = "model.bert."
            model_key = model_key_prefix + key.replace("context_lm.","")
            model_state_dict[key]= ckpt[model_key].clone()
        if "response_lm." in key:
            model_key_prefix = "model.bert."
            model_key = model_key_prefix + key.replace("response_lm.","")
            model_state_dict[key]= ckpt[model_key].clone()
    vocab_size = model.context_lm.config.vocab_size
    model.context_lm.resize_token_embeddings(vocab_size + 1)
    model.response_lm.resize_token_embeddings(vocab_size + 1)
    model.load_state_dict(model_state_dict)
    return model

def load_ckpt_test(model,ckpt_path):
    vocab_size = model.context_lm.config.vocab_size
    # model.context_lm.resize_token_embeddings(vocab_size + 1)
    # model.response_lm.resize_token_embeddings(vocab_size + 1)
    ckpt = torch.load(ckpt_path)["model"]
    model.load_state_dict(ckpt)
    return model