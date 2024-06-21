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


    

def process(device_id, idx, url, output, maxcount=999999999, batch_size=600):
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
        pretrained='/home/user/data/LightClip/CLIP-KD/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin',
        precision="bf16"
    )
    datacomp_clip = datacomp_clip.to(device)


    # Load dataset
    src = wds.WebDataset(url) \
        .decode("pilrgb", handler=log_and_continue) \
        .to_tuple("__key__", "jpg;png;jpeg;webp", "json", "txt", "syn_caps.json", "paug.json", "pth")
    src = src.batched(batch_size)


    with wds.TarWriter(output) as dst:
        
        for batch in tqdm(src, desc=f"Processing Images on GPU {device_id}, Process {idx}"):
            keys, images, jsons, texts, syn_caps,  paugs, t_embs = batch
        
            tokenizer = get_tokenizer(model_name="ViT-L-14")
            text_input = []
            for text, syn_cap_list in zip(texts, syn_caps):
                text = [text]
                text.extend(syn_cap_list)
                text_input.extend(text)

            # text_input = texts + [syn_cap for sublist in syn_caps for syn_cap in sublist]
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
                    "paug.json": paugs[i],
                    "pth": {
                        "image_emb": t_embs[i]['image_emb'],
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
    input_shards = braceexpand("{00128..00129}")#{00000..00830},{00831..01242}
    output_shards = braceexpand("{00128..00129}")
    inputs = [f"/home/user/data/cc12m_dr/{shard}.tar" for shard in input_shards]
    outputs = [f"/home/user/data/cc12m_fixsyn_dr/{shard}.tar" for shard in output_shards]


    with ProcessPoolExecutor(max_workers=num_gpus * models_per_gpu) as executor:
        futures = []
        for i in range(len(inputs)):
            device_id = i % num_gpus
            proc_idx = i % models_per_gpu
            futures.append(executor.submit(process, device_id, proc_idx, inputs[i], outputs[i], batch_size=3000))
        
        for future in tqdm(futures, desc="Total Progress"):
            future.result()

    print('done')

if __name__ == "__main__":
    dr_aug_emb()
