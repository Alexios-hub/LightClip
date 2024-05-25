import logging
import os
import sys
import random
import json
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from open_clip import ClipLoss, get_cast_dtype
from open_clip import KDClipLoss,MultiClipLoss
from open_clip.factory import get_model_config
from open_clip.transform import image_transform
try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms
from open_clip import trace_model, get_tokenizer
from open_clip import AppleMobileCLIP
# from training.data import get_data
from training.distill_data import get_data_distill
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_kd_one_epoch, evaluate
from training.light_swin import _create_lightweight_swin_transformer
from training.light_transformer import LightTransformer

from training.modeling_perceiver_xattn import Perceiver,ParallelPerceiver

import mobileclip
from mobileclip.models.mci import ParallelAttentionBlock,AttentionBlock
from mobileclip.modules.common.transformer import ParallelTransformerEncoder
from open_clip.model import CLIPVisionCfg,CLIPTextCfg
import copy
import torchvision.transforms as transforms

apple_mobile_clip_models = ["mobileclip_s0","mobileclip_s1","mobileclip_s2","mobileclip_b","mobileclip_blt"]

def create_apple_mobile_clip_model(device,mobile_model_name = "mobileclip_s0",pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt'):
    mobile_model, _, _ = mobileclip.create_model_and_transforms(mobile_model_name, pretrained=pretrained)#this preprocess lack convert RGB function
    preprocess = transforms.Compose([
                transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(256, 256)),
                transforms.Lambda(lambda image: image.convert('RGB')),  # Assuming _convert_to_rgb is this
                transforms.ToTensor()
                ])

    cfg = mobile_model.cfg
    vision_cfg = CLIPVisionCfg()
    vision_cfg.image_size = cfg['image_cfg']['image_size']

    text_cfg = CLIPTextCfg()
    text_cfg.context_length = cfg['text_cfg']['context_length']
    text_cfg.vocab_size = cfg['text_cfg']['vocab_size']
    text_cfg.layers = cfg['text_cfg']['n_transformer_layers']

    init_params = {
            "embed_dim":cfg['embed_dim'],
            "vision_cfg":vision_cfg,
            "text_cfg":text_cfg,
            "quick_gelu":False,
            "cast_dtype":None
        }
    
    model = AppleMobileCLIP(**(init_params)).to(device)
    del model.visual
    model.visual = mobile_model.image_encoder.to(device)
    del model.transformer
    model.transformer = mobile_model.text_encoder.to(device)
    del mobile_model
    return model,preprocess,preprocess


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def analyze_model_components(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # 遍历顶层子模块
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if module_params > 0:
            # 计算该子模块的参数占总参数的百分比
            percentage = 100.0 * module_params / total_params
            print(f"{name}: {module_params} parameters, {percentage:.2f}% of total")

def main(args):
    args = parse_args(args)
    print(f'args:{args}')

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')
    try:
        with open(os.path.join(os.getcwd(), 'open_clip/model_configs/'+args.t_model+'.json'), 'r') as f:
            args.t_embed_dim = json.load(f)['embed_dim']
    except Exception as e:
        args.t_embed_dim = -1
    try:
        with open(os.path.join(os.getcwd(), 'open_clip/model_configs/'+args.model+'.json'), 'r') as f:
            args.s_embed_dim = json.load(f)['embed_dim']
    except Exception as e:
        args.s_embed_dim = -1
    
    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"t_model_{args.teachers}",
            f"s_model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"tag_{args.tag}"
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    tokenizers = []
    if args.model in apple_mobile_clip_models:
        model, preprocess_train, preprocess_val = create_apple_mobile_clip_model(device=device,
                                                                                 mobile_model_name=args.model,
                                                                                 pretrained=f'/home/alex/data/LightClip/ml-mobileclip/checkpoints/{args.model}.pt')
        args.s_embed_dim = 512
        tokenizers.append(mobileclip.get_tokenizer(args.model))
    else:
        model, preprocess_train, preprocess_val = create_model_and_transforms(args.model,
                                                                              args.pretrained,
                                                                              precision=args.precision,
                                                                              device=device,
                                                                              jit=args.torchscript,
                                                                              force_quick_gelu=args.force_quick_gelu,
                                                                              force_custom_text=args.force_custom_text,
                                                                              pretrained_image=args.pretrained_image,
                                                                              image_mean=args.image_mean,
                                                                              image_std=args.image_std)
        tokenizers.append(get_tokenizer(args.model))
    preprocess_train = [preprocess_train]
    preprocess_val = [preprocess_val]

    teacher_models = []
    args.t_embed_dim = 0
    for i in range(len(args.teachers)):
        
        teacher = args.teachers[i]
        ckpt_path = args.t_model_checkpoint[i]

        if teacher in apple_mobile_clip_models:
            temp_t_model, temp_preprocess_train, _ = create_apple_mobile_clip_model(device=device,
                                                                                 mobile_model_name=teacher,
                                                                                 pretrained=f'/home/alex/data/LightClip/ml-mobileclip/checkpoints/{teacher}.pt')
            args.t_embed_dim = args.t_embed_dim + 512
            tokenizers.append(mobileclip.get_tokenizer(teacher))
        else:
            temp_t_model, temp_preprocess_train, _ = create_model_and_transforms(teacher,
                                                                              args.pretrained,
                                                                              precision=args.precision,
                                                                              device=device,
                                                                              jit=args.torchscript,
                                                                              force_quick_gelu=args.force_quick_gelu,
                                                                              force_custom_text=args.force_custom_text,
                                                                              pretrained_image=args.pretrained_image,
                                                                              image_mean=args.image_mean,
                                                                              image_std=args.image_std)
            
            with open(os.path.join(os.getcwd(), 'open_clip/model_configs/'+teacher+'.json'), 'r') as f:
                args.t_embed_dim = args.t_embed_dim + json.load(f)['embed_dim']
            for t_n, t_p in temp_t_model.named_parameters():
                t_p.requires_grad = False
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if "state_dict" in checkpoint.keys():
                sd = checkpoint["state_dict"]
            else:
                sd = checkpoint
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            temp_t_model.load_state_dict(sd)
            tokenizers.append(get_tokenizer(teacher))

        temp_t_model.eval()
        if is_master(args):
            logging.info(f'Teacher model {teacher} loaded successfully')
        teacher_models.append(temp_t_model)
        preprocess_train.append(temp_preprocess_train)


    if args.light:
        print(f'light_version:{args.light_version}')
        if args.light_version == "light_swin_tiny":
            del model.visual
            model_cfg = get_model_config(args.model)
            model.visual = _create_lightweight_swin_transformer(variant='light_swin_tiny_patch4_window7_224').to(device=device)
            model.visual.head = torch.nn.Linear(768,model_cfg["embed_dim"]).to(device=device)

            del model.transformer
            model.transformer = LightTransformer(width=model_cfg["text_cfg"]["width"],layers=model_cfg["text_cfg"]["layers"],heads=model_cfg["text_cfg"]["heads"]).to(device=device)
        elif args.light_version == "mobileclip_s0":
            mobile_model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt')
            model = AppleMobileCLIP(**(model.init_params)).to(device)
            del model.visual
            model.visual = mobile_model.image_encoder.to(device)
            model.visual.model.network[7][0] = AttentionBlock(**model.visual.model.network[7][0].init_params).to(device)
            model.visual.model.network[7][1] = AttentionBlock(**model.visual.model.network[7][1].init_params).to(device)
            del model.transformer
            model.transformer = mobile_model.text_encoder.to(device)
            # preprocess_train = transforms.Compose([
            #     transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.CenterCrop(size=(256, 256)),
            #     transforms.Lambda(lambda image: image.convert('RGB')),  # Assuming _convert_to_rgb is this
            #     transforms.ToTensor()
            #     ])
            # preprocess_val = transforms.Compose([
            #     transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.CenterCrop(size=(256, 256)),
            #     transforms.Lambda(lambda image: image.convert('RGB')),  # Assuming _convert_to_rgb is this
            #     transforms.ToTensor()
            #     ])

            del mobile_model
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze specific layers
            for param in model.visual.model.network[7][0].parameters():
                param.requires_grad = True
            for param in model.visual.model.network[7][1].parameters():
                param.requires_grad = True

            # t_mobile_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt')
            # t_model = AppleMobileCLIP(**(t_model.init_params)).to(device)
            # del t_model.visual
            # t_model.visual = t_mobile_model.image_encoder.to(device)

            # del t_model.transformer
            # t_model.transformer = t_mobile_model.text_encoder.to(device)
            # del t_mobile_model

        elif args.light_version == "light_mobileclip_s0":
            mobile_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt')
            model = AppleMobileCLIP(**(model.init_params)).to(device)
            del model.visual
            model.visual = mobile_model.image_encoder.to(device)
            
            model.visual.model.network[7][0] = ParallelAttentionBlock(**model.visual.model.network[7][0].init_params).to(device)
            # model.visual.model.network[7][1] = model.visual.model.network[7][0]
            model.visual.model.network[7][1] = ParallelAttentionBlock(**model.visual.model.network[7][1].init_params).to(device)
            analyze_model_components(model.visual.model.network)

            del model.transformer
            model.transformer = mobile_model.text_encoder.to(device)
            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze specific layers
            for param in model.visual.model.network[7][0].parameters():
                param.requires_grad = True
            for param in model.visual.model.network[7][1].parameters():
                param.requires_grad = True

            del mobile_model


        elif args.light_version == "ws_light_mobileclip_s0":
            mobile_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt')
            model = AppleMobileCLIP(**(model.init_params)).to(device)
            del model.visual
            model.visual = mobile_model.image_encoder.to(device)
            
            model.visual.model.network[7][0] = ParallelAttentionBlock(**model.visual.model.network[7][0].init_params).to(device)

            model.visual.model.network[7][1] = ParallelAttentionBlock(**model.visual.model.network[7][1].init_params).to(device)
            del model.visual.model.network[7][1].token_mixer
            model.visual.model.network[7][1].token_mixer = model.visual.model.network[7][0].token_mixer

            analyze_model_components(model.visual.model.network)

            del model.transformer
            model.transformer = mobile_model.text_encoder.to(device)
            # model.transformer.transformer[1] = ParallelTransformerEncoder(embed_dim=512,ffn_latent_dim=2048,dropout=0.0,ffn_dropout=0.0,stochastic_dropout=0.0).to(device)
            # model.transformer.transformer[2] = ParallelTransformerEncoder(embed_dim=512,ffn_latent_dim=2048,dropout=0.0,ffn_dropout=0.0,stochastic_dropout=0.0).to(device)
            # model.transformer.transformer[3] = ParallelTransformerEncoder(embed_dim=512,ffn_latent_dim=2048,dropout=0.0,ffn_dropout=0.0,stochastic_dropout=0.0).to(device)
            # model.transformer.transformer[4] = ParallelTransformerEncoder(embed_dim=512,ffn_latent_dim=2048,dropout=0.0,ffn_dropout=0.0,stochastic_dropout=0.0).to(device)

            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze specific layers
            for param in model.visual.model.network[7][0].parameters():
                param.requires_grad = True
            for param in model.visual.model.network[7][1].parameters():
                param.requires_grad = True

            # for param in model.transformer.transformer[1].parameters():
            #     param.requires_grad = True
            # for param in model.transformer.transformer[2].parameters():
            #     param.requires_grad = True
            # for param in model.transformer.transformer[3].parameters():
            #     param.requires_grad = True
            # for param in model.transformer.transformer[4].parameters():
            #     param.requires_grad = True

            del mobile_model

        elif args.light_version == "perceiver_mobileclip_s0":
            mobile_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt')
            model = AppleMobileCLIP(**(model.init_params)).to(device)
            del model.visual
            model.visual = mobile_model.image_encoder.to(device)

            init_params_0 = model.visual.model.network[7][0].init_params
            model.visual.model.network[7][0] = Perceiver(dim=init_params_0['dim'],k_v_dim=init_params_0['dim'],depth=1,ff_mult=init_params_0['mlp_ratio']).to(device)
            del model.visual.model.network[7][1]

            analyze_model_components(model.visual.model.network)

            del model.transformer
            model.transformer = mobile_model.text_encoder.to(device)
            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze specific layers
            for param in model.visual.model.network[7][0].parameters():
                param.requires_grad = True

            del mobile_model
        
        elif args.light_version == "light_perceiver_mobileclip_s0":
            mobile_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/alex/data/LightClip/ml-mobileclip/checkpoints/mobileclip_s0.pt')
            model = AppleMobileCLIP(**(model.init_params)).to(device)
            del model.visual
            model.visual = mobile_model.image_encoder.to(device)

            init_params_0 = model.visual.model.network[7][0].init_params
            model.visual.model.network[7][0] = ParallelPerceiver(dim=init_params_0['dim'],k_v_dim=init_params_0['dim'],depth=1,ff_mult=init_params_0['mlp_ratio']).to(device)
            del model.visual.model.network[7][1]

            analyze_model_components(model.visual.model.network)

            del model.transformer
            model.transformer = mobile_model.text_encoder.to(device)
            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze specific layers
            for param in model.visual.model.network[7][0].parameters():
                param.requires_grad = True

            del mobile_model

        else:
            raise KeyError(f'{args.light_version} not supported.')
    if is_master(args):
        logging.info(f'model:{model}')
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        for idx, t_model in enumerate(teacher_models):
            teacher_name = args.teachers[idx]  
            logging.info(f"Teacher Visual Params ({teacher_name}):")
            logging.info(f"{str(sum([i.numel() for i in t_model.visual.parameters()])/1e6)}M")
            logging.info(f"Teacher Text Params ({teacher_name}):")
            logging.info(f"{str(sum([i.numel() for i in t_model.transformer.parameters()])/1e6)}M")

        logging.info("Student Visual Params:")
        logging.info(f"{str(sum([i.numel() for i in model.visual.parameters()])/1e6)}M")
        logging.info("Student Text Params:")
        logging.info(f"{str(sum([i.numel() for i in model.transformer.parameters()])/1e6)}M")

        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")


    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    
    loss = MultiClipLoss(
        args=args,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod).cuda()
    
    # create optimizer and scaler
    optimizer = None
    scaler = None
    
    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
                {"params": loss.parameters()}
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    # data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    data = get_data_distill(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=tokenizers)
    assert len(data), 'At least one train or eval dataset must be specified.'

    
    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    if args.t_eval:
        print('evaluate teacher:')
        evaluate(t_model, data, start_epoch, args, writer)# todo:evaluate teachers
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')) and epoch == start_epoch:
            evaluate(model, data, epoch, args, writer)

        if epoch == 5 and (args.light_version == "light_mobileclip_s0" or args.light_version == "ws_light_mobileclip_s0"):#unfreeze modules top of attention block at epoch 5
            if is_master(args):
                logging.info("unfreeze proj module of image enc.")
            for param in model.module.visual.model.conv_exp.parameters():
                param.requires_grad = True
            for param in model.module.visual.model.head.parameters():
                param.requires_grad = True

            # for param in model.module.transformer.transformer[5].parameters():#unfreeze modules top of transformer encoder at epoch 5
            #     param.requires_grad = True

        train_kd_one_epoch(model, teacher_models, data, epoch, loss, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    logging.info(f'The files are saved at {args.logs}')
    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
