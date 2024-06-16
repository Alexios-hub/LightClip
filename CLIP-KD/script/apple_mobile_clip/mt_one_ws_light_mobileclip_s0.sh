cd src
torchrun --nproc_per_node 1 -m \
    --master_addr=127.0.0.2 --master_port=29566 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/home/alex/data/LightClip/datasets/DataComp_small/shards/{00000000..00001287}.tar' \
    --dataset-type webdataset \
    --train-num-samples 11369022\
    --imagenet-val=/home/alex/data/LightClip/datasets/val_images/ \
    --warmup 6000 \
    --batch-size=1024 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs 8 \
    --workers=1 \
    --model mobileclip_s0 \
    --teachers mobileclip_s0\
    --t-model-checkpoint /home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt \
    --resume /home/alex/data/LightClip/CLIP-KD/logs/2024_06_06-15_05_23-t_model_['mobileclip_s0']-s_model_mobileclip_s0-lr_1e-05-b_1024-tag_distill-new/checkpoints/epoch_1.pt \
    --logs /home/alex/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 0. \
    --alpha_fd_loss 0. \
    --tag distill-new \
    --light \
    --light_version ws_light_mobileclip_s0