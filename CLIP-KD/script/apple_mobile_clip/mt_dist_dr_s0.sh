cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29588 \
    training.main_kd_dr \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/home/alex/data/cc12m_fixsyn_dr/{00000..01242}.tar' \
    --dataset-type webdataset \
    --train-num-samples 12000000\
    --imagenet-val=/home/alex/data/LightClip/datasets/val_images/ \
    --warmup 10000 \
    --batch-size=1024 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 64 \
    --workers=8 \
    --model mobileclip_s0 \
    --logs /home/alex/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 4000. \
    --tag distill-new \
    --light \
    --light_version ws_light_mobileclip_s0 \
    --precision fp32 \