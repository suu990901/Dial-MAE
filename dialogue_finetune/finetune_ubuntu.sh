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