cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29588 \
    training.main_kd_dr \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/home/user/data/cc12m_fixsyn_dr/{00000..01242}.tar' \
    --dataset-type webdataset \
    --train-num-samples 12000000\
    --imagenet-val=/home/user/data/LightClip/datasets/val_images/ \
    --warmup 10000 \
    --batch-size=640 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model mobileclip_s0 \
    --logs /home/user/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 2.5 \
    --alpha_icl_loss 0. \
    --alpha_fd_loss 0. \
    --tag distill-new \
    --light \
    --light_version ws_light_mobileclip_s0 \
    --resume /home/user/data/LightClip/CLIP-KD/logs/2024_06_22-22_34_15-model_mobileclip_s0-lr_0.001-b_640-epochs_32-tag_distill-new/checkpoints/epoch_8.pt\
    --precision fp32 \