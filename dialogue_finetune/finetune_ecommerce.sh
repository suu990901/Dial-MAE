model_path=$1
name=ecommerce
resFile=`basename ${model_path}`
mkdir -p log/$name/$resFile
nohup python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --master_port=23455 main.py \
--train_path data/ecommerce/train/data.json \
--val_path data/ecommerce/val/data.json \
--tensorboard_dir tflog/$name/$resFile \
--dataset ecommerce \
--num_eval_times 50 \
--model $model_path \
--save_path model/$name/$resFile \
--epochs 2 \
--eval_mode val \
--lr 1e-4 \
--batch_size 128 \
>log/$name/$resFile.log 2>&1 &