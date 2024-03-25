
import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers import TrainingArguments

class ProjectionMLP(nn.Module):
    def __init__(self, size):
        super().__init__()
        in_dim = size
        hidden_dim = size * 2
        out_dim = size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class Similarity(nn.Module):
        """
        Dot product or cosine similarity
        """
        def __init__(self, temp):
            super().__init__()
            self.temp = temp
            self.cos = nn.CosineSimilarity(dim=-1)

        def forward(self, x, y):
            return self.cos(x, y) / self.temp

class dialForPretraining(nn.Module):
    def __init__(
        self,
        context_lm: BertModel,
        response_lm: BertModel,
        args
    ):
        super(dialForPretraining, self).__init__()
        self.context_lm = context_lm
        self.response_lm = response_lm
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.mlp = ProjectionMLP(size=768)
        # self.sim = Similarity(args.temp)
        self.model_args = args

    
    def cl_forward(self,model_input):
        device = self.context_lm.device
        context_input = {"input_ids":model_input["context_input_ids"].to(device),
                        "attention_mask":model_input["context_attention_mask"].to(device)}
        response_input = {"input_ids":model_input["response_input_ids"].to(device),
                        "attention_mask":model_input["response_attention_mask"].to(device)}

        context_out = self.context_lm(
            **context_input,
            output_hidden_states=True,
            return_dict=True
        )
        response_out = self.response_lm(
            **response_input,
            output_hidden_states=True,
            return_dict=True
        )
        context_cls_hiddens = context_out.hidden_states[-1][:, 0]
        response_cls_hiddens = response_out.hidden_states[-1][:, 0]

        # response_cls_hiddens=response_cls_hiddens.detach()
        # z1,z2 = self.mlp(context_cls_hiddens),self.mlp(response_cls_hiddens)
        z1,z2 = context_cls_hiddens,response_cls_hiddens
        
        if self.model_args.use_cross_cl and dist.is_initialized():
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
 
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        # sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        sim = torch.matmul(z1,z2.t())
        labels = torch.arange(sim.size(0)).long().to(sim.device)
        loss = self.cross_entropy(sim,labels)
        return loss

    def sentemb_forward(self,model_input,mode='val'):
        if self.model_args.dataset == "ubuntu" or mode=="test":
            d = 10
        else:
            d = 2
        device = self.context_lm.device
        batch_size,_ = model_input["context_input_ids"].shape
        context_index = list(range(0,batch_size,d))
        context_input = {"input_ids":model_input["context_input_ids"][context_index].to(device),
                        "attention_mask":model_input["context_attention_mask"][context_index].to(device)}
        response_input = {"input_ids":model_input["response_input_ids"].to(device),
                        "attention_mask":model_input["response_attention_mask"].to(device)}
        labels = model_input["labels"].to(device)
        labels = labels.view(-1,d)
        context_out = self.context_lm(
            **context_input,
            output_hidden_states=True,
            return_dict=True
        )
        response_out = self.response_lm(
            **response_input,
            output_hidden_states=True,
            return_dict=True
        )
     
        context_cls_hiddens = context_out.hidden_states[-1][:, 0]
        response_cls_hiddens = response_out.hidden_states[-1][:, 0]
        # context_cls_hiddens,response_cls_hiddens=self.mlp(context_cls_hiddens),self.mlp(response_cls_hiddens)
        context_cls_hiddens=context_cls_hiddens.unsqueeze(1)
        response_cls_hiddens=response_cls_hiddens.view(-1,d,768)
        return context_cls_hiddens,response_cls_hiddens,labels

    def forward(self, model_input,sent_emb=False,eval_mode="val"):
        if sent_emb == False:
            loss = self.cl_forward(model_input)
            return loss
        else:
            context_cls_hiddens,response_cls_hiddens,labels = self.sentemb_forward(model_input,eval_mode)
            return context_cls_hiddens,response_cls_hiddens,labels


    @classmethod
    def from_pretrained(
            cls, args,
    ):
        context_lm = BertModel.from_pretrained(args.model)
        response_lm = BertModel.from_pretrained(args.model)
        model = cls(context_lm,response_lm,args)
        return model