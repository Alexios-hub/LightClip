cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29533 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/home/alex/data/LightClip/datasets/cc3m/cc3m/Train_GCC-training.csv"  \
    --val-data="/home/alex/data/LightClip/datasets/cc3m/cc3m/Validation_GCC-1.1.0-Validation.csv"  \
    --data-root /home/alex/data/LightClip/datasets/cc3m/images/ \
    --val-data-root /home/alex/data/LightClip/datasets/cc3m/images/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/home/alex/data/LightClip/datasets/val_images/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model timm-swin_tiny_patch4_window7_224 \
    --t-model ViT-B-16 \
    --t-model-checkpoint /home/alex/data/LightClip/CLIP-KD/pretrained_models/ViT_B_16_cc3m_12m_ep32.pt \
    --logs /home/alex/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 2000. \
    --tag distill-new \
    # --light