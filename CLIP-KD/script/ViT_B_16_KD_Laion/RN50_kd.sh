cd src
torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.2 --master_port=29533 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="path/to/cc3m_train.csv,path/to/cc12m.csv"  \
    --val-data="path/to/cc3m_val.csv"  \
    --data-root path/to/cc3m/images/,path/to/cc12m/images/ \
    --val-data-root path/to/cc3m/images/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=path/to/imagenet/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model RN50 \
    --t-model ViT-B-16 \
    --t-model-checkpoint path/to/vit_b_16-laion400m_e32-55e67d44.pt \
    --logs path/to/logs/ \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 2000. \
    --tag distill-new 