cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29588 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/home/user/data/LightClip/datasets/DataComp_small/shards/{00000000..00001287}.tar' \
    --dataset-type webdataset \
    --train-num-samples 11369022\
    --imagenet-val=/home/user/data/LightClip/datasets/val_images/ \
    --warmup 6000 \
    --batch-size=512 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs 40 \
    --workers=8 \
    --model mobileclip_s0 \
    --teachers ViT-L-14 ViT-L-14-336\
    --t-model-checkpoint /home/user/data/LightClip/CLIP-KD/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin /home/user/data/LightClip/CLIP-KD/pretrained_models/vit_large_patch14_clip_336.openai/open_clip_pytorch_model.bin\
    --resume /home/user/data/LightClip/CLIP-KD/logs/model_on_cc3m_ws/checkpoints/epoch_32.pt \
    --logs /home/user/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 0. \
    --alpha_fd_loss 0. \
    --tag distill-new \
    --light \
    --light_version ws_light_mobileclip_s0 \