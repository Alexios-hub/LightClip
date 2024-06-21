import os
from typing import List
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import open_clip
import torch
import torchvision.transforms as transforms
import webdataset as wds
from tqdm import tqdm
import json

import torch
from torchvision import transforms
import webdataset as wds
from training.data import log_and_continue
from braceexpand import braceexpand

from concurrent.futures import ProcessPoolExecutor
from torch import Tensor
from torchvision.transforms import RandAugment,RandomResizedCrop, InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import _apply_op
from open_clip import create_model_and_transforms, get_tokenizer
import gzip
from io import BytesIO


    
class myRandAugment(RandAugment):
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None):
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)

    def forward(self, img):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        augment_params = []
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            augment_params.append((op_index, magnitude))

        return img, augment_params
    


class myRandomResizedCrop(RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), [i, j, h, w]

    

def process(device_id, idx, url, output, maxcount=999999999, batch_size=400):
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    
    # Initialize models
    openai_clip, openai_preprocess_train, openai_preprocess_val = create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained='openai',
        precision="bf16"
    )
    openai_clip = openai_clip.to(device)
    
    datacomp_clip, datacomp_preprocess_train, datacomp_preprocess_val = create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained='/home/alex/data/LightClip/CLIP-KD/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin',
        precision="bf16"
    )
    datacomp_clip = datacomp_clip.to(device)

    preprocess = openai_preprocess_val

    # Load dataset
    src = wds.WebDataset(url) \
        .decode("pilrgb", handler=log_and_continue) \
        .to_tuple("__key__", "jpg;png;jpeg;webp", "json", "txt", "syn_caps.json")
    src = src.batched(batch_size)

    random_crop = myRandomResizedCrop(size=(224, 224))
    rand_augment = myRandAugment()

    with wds.TarWriter(output) as dst:
        
        for batch in tqdm(src, desc=f"Processing Images on GPU {device_id}, Process {idx}"):
            keys, images, jsons, texts, syn_caps = batch
        
            # Prepare batches
            augmented_images_batch = []
            all_augmentation_params = []
            for image in images:
                for _ in range(5):
                    cropped_img, crop_params = random_crop(image)
                    augmented_img, augment_params = rand_augment(cropped_img)
                    augmented_images_batch.append(preprocess(augmented_img))
                    all_augmentation_params.append([crop_params, augment_params])

            augmented_images_batch = torch.stack(augmented_images_batch).to(device).to(torch.bfloat16)
        
            # Batch inference for images
            with torch.no_grad():
                openai_embs = openai_clip.encode_image(augmented_images_batch).cpu().detach()
                datacomp_embs = datacomp_clip.encode_image(augmented_images_batch).cpu().detach()
        
            image_embs = [torch.cat((openai_emb, datacomp_emb), dim=-1) for openai_emb, datacomp_emb in zip(openai_embs, datacomp_embs)]
        
            tokenizer = get_tokenizer(model_name="ViT-L-14")
            text_input = texts + [syn_cap for sublist in syn_caps for syn_cap in sublist]
            text_input_batch = tokenizer(text_input).to(device)
        
            # Batch inference for texts
            with torch.no_grad():
                openai_text_embs = openai_clip.encode_text(text_input_batch).cpu().detach()
                datacomp_text_embs = datacomp_clip.encode_text(text_input_batch).cpu().detach()

            text_embs = [torch.cat((openai_text_emb,datacomp_text_emb),dim=-1) for openai_text_emb, datacomp_text_emb in zip(openai_text_embs, datacomp_text_embs)]
        

            for i, key in enumerate(keys):
                sample = {
                    "__key__": key,
                    "jpg": images[i],
                    "syn_caps.json": json.dumps(syn_caps[i]),
                    "json": jsons[i],
                    "txt": texts[i],
                    "paug.json": json.dumps(all_augmentation_params[i*5:(i+1)*5]),
                    "pth": {
                        "image_emb": image_embs[i*5:(i+1)*5],
                        "text_emb": text_embs[i*4:(i+1)*4]
                    }
                }
                dst.write(sample)
    del openai_clip
    del datacomp_clip
    torch.cuda.empty_cache()

def dr_aug_emb():
    num_gpus = 2  # 有两个GPU
    models_per_gpu = 1  # 每个GPU运行1个模型实例

    input_shards = braceexpand("{00255..00256}")#{00000..00830},{00831..01242}
    output_shards = braceexpand("{00255..00256}")
    inputs = [f"/home/user/data/cc12m_sync/{shard}.tar" for shard in input_shards]
    outputs = [f"/home/user/data/cc12m_dr/{shard}.tar" for shard in output_shards]



    with ProcessPoolExecutor(max_workers=num_gpus * models_per_gpu) as executor:
        futures = []
        for i in range(len(inputs)):
            device_id = i % num_gpus
            proc_idx = i % models_per_gpu
            futures.append(executor.submit(process, device_id+2, proc_idx, inputs[i], outputs[i], batch_size=300))



    with ProcessPoolExecutor(max_workers=num_gpus * models_per_gpu) as executor:
        futures = []
        for i in range(len(inputs)):
            device_id = i % num_gpus
            proc_idx = i % models_per_gpu
            futures.append(executor.submit(process, device_id, proc_idx, inputs[i], outputs[i], batch_size=600))

        
        for future in tqdm(futures, desc="Total Progress"):
            future.result()

    print('done')

if __name__ == "__main__":
    dr_aug_emb()
