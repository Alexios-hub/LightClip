import os
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


def process(model_name, pretrained, device_id, idx, url, output, maxcount=999999999, batch_size=400):
    torch.cuda.set_device(device_id)  # 明确设置进程的GPU
    device = torch.device(f'cuda:{device_id}')
    model, _, base_transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained
    )
    model = model.to(device)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        base_transform
    ])
    src = wds.WebDataset(url) \
        .decode("pilrgb", handler=log_and_continue) \
        .to_tuple("__key__", "jpg;png;jpeg;webp", "json", "txt")
    
    src = src.batched(batch_size)

    with wds.TarWriter(output) as dst:
        for batch in tqdm(src, desc=f"Processing Images on GPU {device_id}, Process {idx}"):
            keys, images, jsons, texts = batch
            input_images = [transform(image).unsqueeze(0).to(device) for image in images]
            input_images = torch.cat(input_images, dim=0)  # 将所有图像合并为一个批处理
            input_images = torch.repeat_interleave(input_images, repeats=3, dim=0)  # 每个图像复制3次

            with torch.no_grad(), torch.cuda.amp.autocast():
                generated = model.generate(input_images, generation_type='top_p', top_p=0.5, temperature=0.9)

            # 处理每个原始图像的3个captions
            syn_caps = []
            for i in range(0, generated.shape[0], 3):
                captions = [open_clip.decode(generated[j]).split("<end_of_text>")[0].replace("<start_of_text>", "")
                            for j in range(i, i+3)]
                syn_caps.append(captions)

            for key, image, syn_cap_group, data, text in zip(keys, images, syn_caps, jsons, texts):
                sample = {
                    "__key__": key,
                    "jpg": image,
                    "syn_caps.json": json.dumps(syn_cap_group),
                    "json": data,
                    "txt": text
                }
                dst.write(sample)
    del model
    torch.cuda.empty_cache()

def aug_syn_cap():
    model_name = "coca_ViT-L-14"
    pretrained = "mscoco_finetuned_laion2B-s13B-b90k"
    num_gpus = 2  # 有两个GPU
    models_per_gpu = 1  # 每个GPU运行1个模型实例
    input_shards = braceexpand("{00125..00127}")
    output_shards = braceexpand("{00125..00127}")
    inputs = [f"/home/user/data/cc12m/{shard}.tar" for shard in input_shards]
    outputs = [f"/home/user/data/cc12m_sync/{shard}.tar" for shard in output_shards]
    
    with ProcessPoolExecutor(max_workers=num_gpus * models_per_gpu) as executor:
        futures = []
        for i in range(len(inputs)):
            device_id = i % num_gpus
            proc_idx = i % models_per_gpu
            futures.append(executor.submit(process, model_name, pretrained, device_id, proc_idx, inputs[i], outputs[i]))
        
        for future in tqdm(futures, desc="Total Progress"):
            future.result()

    print('done')

if __name__ == "__main__":
    aug_syn_cap()


    

