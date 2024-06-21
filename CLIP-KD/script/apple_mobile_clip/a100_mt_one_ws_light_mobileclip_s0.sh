# cd src
# torchrun --nproc_per_node 2 -m \
#     --master_addr=127.0.0.2 --master_port=29566 \
#     training.main_kd \
#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --report-to tensorboard \
#     --train-data="/home/user/data/LightClip/datasets/cc3m/cc3m/Train_GCC-training.csv"  \
#     --val-data="/home/user/data/LightClip/datasets/cc3m/cc3m/Validation_GCC-1.1.0-Validation.csv"  \
#     --data-root /home/user/data/LightClip/datasets/cc3m/images/ \
#     --val-data-root /home/user/data/LightClip/datasets/cc3m/images/ \
#     --csv-img-key filepath \
#     --csv-caption-key title \
#     --imagenet-val=/home/user/data/LightClip/datasets/val_images/ \
#     --warmup 10000 \
#     --batch-size=512 \
#     --lr=1e-3 \
#     --wd=0.1 \
#     --epochs 32 \
#     --workers=8 \
#     --model mobileclip_s0 \
#     --teachers mobileclip_s0 ViT-L-14\
#     --t-model-checkpoint /home/user/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt /home/user/data/LightClip/CLIP-KD/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin\
#     --logs /home/user/data/LightClip/CLIP-KD/logs \
#     --alpha_ckd_loss 1. \
#     --alpha_icl_loss 1. \
#     --alpha_fd_loss 2000. \
#     --tag distill-new \
#     --light \
#     --light_version ws_light_mobileclip_s0 \

cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29566 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/home/user/data/LightClip/datasets/cc12m/{00000..01242}.tar' \
    --dataset-type webdataset \
    --train-num-samples 8500000\
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/home/user/data/LightClip/datasets/val_images/ \
    --warmup 10000 \
    --batch-size=1024 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model mobileclip_s0 \
    --teachers mobileclip_s0\
    --t-model-checkpoint /home/user/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt \
    --logs /home/user/data/LightClip/CLIP-KD/logs \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 2000. \
    --tag distill-new \
    --light \
    --light_version ws_light_mobileclip_s0
