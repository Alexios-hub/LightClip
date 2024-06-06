cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29566 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/home/user/data/LightClip/datasets/DataComp_small/shards/{00000000..00001287}.tar' \
    --dataset-type webdataset \
    --train-num-samples 11369022\
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/home/user/data/LightClip/datasets/val_images/ \
    --warmup 10000 \
    --batch-size=512 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model mobileclip_s0 \
    --resume /home/user/data/LightClip/CLIP-KD/logs/model_on_cc3m_ws/checkpoints/epoch_32.pt \
    --teachers mobileclip_s0\
    --t-model-checkpoint /home/user/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt \
    --logs /home/user/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 2000. \
    --tag distill-new \
    --light \
    --light_version ws_light_mobileclip_s0