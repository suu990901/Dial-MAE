# Pre-training

## Data Preparation
Download the original versions of the datasets separately from [Ubuntu Corpus V1](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip) and [E-Commerce Corpus](https://github.com/cooelf/DeepUtteranceAggregation). Then, run the following example script to perform data preprocessing.

```
bash make_finetune_data.sh
```

## Fine-tuning

The code below will launch fine-tuing on 8 GPUs and train Dial-MAE with warm start from bert-base-uncased using the Ubuntu Corpus.

```
model_path=$1
name=ubuntu
resFile=`basename ${model_path}`
mkdir -p log/$name/$resFile
nohup python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --master_port=23455 main.py \
--train_path data/ubuntu/train/data.json \
--val_path data/ubuntu/val/data.json \
--tensorboard_dir tflog/$name/$resFile \
--dataset ubuntu \
--num_eval_times 50 \
--model $model_path \
--save_path model/$name/$resFile \
--epochs 5 \
--eval_mode val \
--lr 5e-5 \
--batch_size 64 \
>log/$name/$resFile.log 2>&1 &
```
## Post-training Examples
You can also run the following script to post-train separately on Ubuntu and Ecommerce.

```
bash finetune_ubuntu.sh
bash finetune_ecommerce.sh
```
