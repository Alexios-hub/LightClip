import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_image_features(
        image_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'    
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
        
        if not local_loss:
            # Ensure grads for local rank when all_image_features don't have a gradient
            gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
            gathered_image_features[rank] = image_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
    else:
        if gather_with_grad:
            # Collect tensors from all processes
            all_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            dist.all_gather(all_image_features, image_features)
            all_image_features = torch.cat(all_image_features, dim=0)
        else:
            # No gradient computation needed, just gather
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            
            if not local_loss:
                # Ensure grads for local rank when all_image_features don't have a gradient
                gathered_image_features[rank] = image_features
            all_image_features = torch.cat(gathered_image_features, dim=0)

    return all_image_features


def gather_text_features(
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'    
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        
        if gather_with_grad:
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_text_features = hvd.allgather(text_features)
        
        if not local_loss:
            # Ensure grads for local rank when all_text_features don't have a gradient
            gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
            gathered_text_features[rank] = text_features
            all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        if gather_with_grad:
            # Collect tensors from all processes
            all_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(all_text_features, text_features)
            all_text_features = torch.cat(all_text_features, dim=0)
        else:
            # No gradient computation needed, just gather
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_text_features, text_features)
            
            if not local_loss:
                # Ensure grads for local rank when all_text_features don't have a gradient
                gathered_text_features[rank] = text_features
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_text_features

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def sim_stats(all_image_features, all_text_features, logits_per_image, logits_per_text, labels, num_logits):
    img_num = all_image_features.size(0)
    mask = torch.eye(img_num).cuda()
    img_to_img_sim = ((all_image_features @ all_image_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    txt_to_txt_sim = ((all_text_features @ all_text_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    img_to_img_nn_sim = ((all_image_features @ all_image_features.T) * (1 - mask)-2 * mask).detach().max(dim=1)[0].mean()
    txt_to_txt_nn_sim = ((all_text_features @ all_text_features.T) * (1 - mask)-2 * mask).detach().max(dim=1)[0].mean()
    img_to_pos_txt_sim = ((all_image_features @ all_text_features.T) * mask).detach().sum() / img_num
    img_to_neg_txt_sim = ((all_image_features @ all_text_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    img_to_hard_neg_txt_sim = (((all_image_features @ all_text_features.T) * (1 - mask)+ mask * (-2.)).max(dim=1))[0].mean()
    img_to_pos_minus_neg_txt_sim = img_to_pos_txt_sim - img_to_neg_txt_sim
    img_to_pos_minus_hard_neg_txt_sim = img_to_pos_txt_sim - img_to_hard_neg_txt_sim
    
    txt_to_pos_img_sim = ((all_text_features @ all_image_features.T) * mask).detach().sum() / img_num
    txt_to_neg_img_sim = ((all_text_features @ all_image_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    txt_to_hard_neg_img_sim = (((all_text_features @ all_image_features.T) * (1 - mask)+ mask * (-2.)).max(dim=1))[0].mean()
    txt_to_pos_minus_neg_img_sim = txt_to_pos_img_sim - txt_to_neg_img_sim
    txt_to_pos_minus_hard_neg_img_sim = txt_to_pos_img_sim - txt_to_hard_neg_img_sim
    
    targets = F.one_hot(labels, num_classes=logits_per_image.size(1)).float()
    prob_per_image = F.softmax(logits_per_image, 1)
    neg_prob_per_image = prob_per_image.clone()
    neg_prob_per_image[mask.bool()] = 0.
    hard_neg_indices = F.one_hot(neg_prob_per_image.max(dim=1)[1], num_classes=num_logits)
    img_anchor_grad_from_hard_neg = F.normalize((neg_prob_per_image * hard_neg_indices) @ all_text_features, dim=1)
    img_anchor_grad_from_neg = F.normalize(neg_prob_per_image @ all_text_features, dim=1)
    img_anchor_grad_from_pos = F.normalize((prob_per_image * mask - mask) @ all_text_features, dim=1)

    img_anchor_txt_pos_neg_grad_sim = (img_anchor_grad_from_pos * img_anchor_grad_from_neg).sum(dim=1).mean()
    img_anchor_txt_pos_hard_neg_grad_sim = (img_anchor_grad_from_pos * img_anchor_grad_from_hard_neg).sum(dim=1).mean()
    
    prob_per_txt = F.softmax(logits_per_text, 1)
    neg_prob_per_txt = prob_per_txt.clone()
    neg_prob_per_txt[mask.bool()] = 0.
    hard_neg_txt_indices = F.one_hot(neg_prob_per_txt.max(dim=1)[1], num_classes=num_logits)
    txt_anchor_grad_from_hard_neg = F.normalize((neg_prob_per_txt * hard_neg_txt_indices) @ all_image_features, dim=1)
    txt_anchor_grad_from_neg = F.normalize(neg_prob_per_txt @ all_image_features, dim=1)
    txt_anchor_grad_from_pos = F.normalize((prob_per_txt * mask - mask) @ all_image_features, dim=1)

    txt_anchor_img_pos_neg_grad_sim = (txt_anchor_grad_from_pos * txt_anchor_grad_from_neg).sum(dim=1).mean()
    txt_anchor_img_pos_hard_neg_grad_sim = (txt_anchor_grad_from_pos * txt_anchor_grad_from_hard_neg).sum(dim=1).mean()

    
    sims = [img_to_img_sim.item(), 
            txt_to_txt_sim.item(), 
            img_to_pos_txt_sim.item(), 
            img_to_neg_txt_sim.item(), 
            img_to_hard_neg_txt_sim.item(),
            img_to_pos_minus_neg_txt_sim.item(), 
            img_to_pos_minus_hard_neg_txt_sim.item(),
            txt_to_pos_img_sim.item(),
            txt_to_neg_img_sim.item(),
            txt_to_hard_neg_img_sim.item(),
            txt_to_pos_minus_neg_img_sim.item(),
            txt_to_pos_minus_hard_neg_img_sim.item(),
            img_anchor_txt_pos_neg_grad_sim.item(),
            img_anchor_txt_pos_hard_neg_grad_sim.item(),
            txt_anchor_img_pos_neg_grad_sim.item(),
            txt_anchor_img_pos_hard_neg_grad_sim.item(),
            img_to_img_nn_sim.item(),
            txt_to_txt_nn_sim.item(),]

    return sims


def get_grad(p, k, tau, targets):
    logits = p @ k.T / tau
    targets = F.one_hot(targets, num_classes=logits.size(1)).float()
    prob = F.softmax(logits, 1)
    grad_p = (prob - targets) @ k / tau / targets.size(0)
    embed_size = p.size(1)
    prob_targets_repeat = (prob - targets).t().repeat(1, embed_size).view(-1,embed_size, p.size(0))
    grad_k = (prob_targets_repeat * (p.t() / tau).unsqueeze(0)).sum(-1) / targets.size(0)

    return grad_p, grad_k
     
    
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        return total_loss


    
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

class KDClipLoss(nn.Module):

    def __init__(
            self,
            args,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.args = args

        if args.t_embed_dim != args.s_embed_dim:
            self.visual_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
            self.text_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
        
        if args.alpha_afd_loss > 0.:
            self.visual_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
            self.text_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
            
        # cache state
        self.prev_num_logits = 0
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, \
        t_image_features, t_text_features, t_logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            t_all_image_features, t_all_text_features = gather_features(
                t_image_features, t_text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            t_logits_per_image = t_logit_scale * t_all_image_features @ t_all_text_features.T
            t_logits_per_text = t_logits_per_image.T

            normalized_image_features = F.normalize(image_features, dim=1)
            normalized_text_features = F.normalize(text_features, dim=1)
            normalized_all_image_features = F.normalize(all_image_features, dim=1)
            normalized_all_text_features = F.normalize(all_text_features, dim=1)
            
            if self.local_loss:
                logits_per_image = logit_scale * normalized_image_features @ normalized_all_text_features.T
                logits_per_text = logit_scale * normalized_text_features @ normalized_all_image_features.T
            else:
                logits_per_image = logit_scale * normalized_all_image_features @ normalized_all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            all_image_features = image_features
            all_text_features = text_features
            t_all_image_features = t_image_features
            t_all_text_features = t_text_features

            t_logits_per_image = t_logit_scale * t_all_image_features @ t_all_text_features.T
            t_logits_per_text = t_logits_per_image.T

            normalized_image_features = F.normalize(image_features,dim=-1)
            normalized_text_features = F.normalize(text_features,dim=-1)
            logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.T
            logits_per_text = logit_scale * normalized_text_features @ normalized_image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.args.t_embed_dim != self.args.s_embed_dim:
            all_image_features = self.visual_proj(all_image_features)
            all_text_features = self.text_proj(all_text_features)
            
        normalized_all_image_features = F.normalize(all_image_features, dim=1)
        normalized_all_text_features = F.normalize(all_text_features, dim=1)
        fd_loss = F.mse_loss(normalized_all_image_features, t_all_image_features) +\
            F.mse_loss(normalized_all_text_features, t_all_text_features)
            
        logits_per_s_image_to_t_text = self.cross_logit_scale * normalized_all_image_features @ t_all_text_features.T
        logits_per_s_text_to_t_image = self.cross_logit_scale * normalized_all_text_features @ t_all_image_features.T
        
        task_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        ckd_loss = torch.tensor(0.).cuda() 
        icl_loss = torch.tensor(0.).cuda() 
        cross_kd_loss = torch.tensor(0.).cuda() 
        gd_loss = torch.tensor(0.).cuda() 
        afd_loss = torch.tensor(0.).cuda() 
        
        icl_loss = (
            F.cross_entropy(logits_per_s_image_to_t_text, labels) +
            F.cross_entropy(logits_per_s_text_to_t_image, labels)
            ) / 2
        
        ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2
        
        cross_kd_loss = (self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2
        #kd_loss = (F.cross_entropy(logits_per_image, F.softmax(, dim=1)) \
        #    + F.cross_entropy(logits_per_text, F.softmax(t_logits_per_text.detach(), dim=1))) / 2
        
        
        if self.args.alpha_gd_loss > 0.:
            with torch.no_grad():
                t_grad_p_img, t_grad_k_txt = get_grad(t_all_image_features, t_all_text_features, t_logit_scale, labels)
                t_grad_p_txt, t_grad_k_img = get_grad(t_all_text_features, t_all_image_features, t_logit_scale, labels)
            
            s_grad_p_img, s_grad_k_txt = get_grad(normalized_all_image_features, normalized_all_text_features, logit_scale, labels)
            s_grad_p_txt, s_grad_k_img = get_grad(normalized_all_text_features, normalized_all_image_features, logit_scale, labels)

            gd_loss = F.mse_loss(s_grad_p_img, t_grad_p_img.detach()) +\
                F.mse_loss(s_grad_k_txt, t_grad_k_txt.detach()) +\
                    F.mse_loss(s_grad_p_txt, t_grad_p_txt.detach()) +\
                        F.mse_loss(s_grad_k_img, t_grad_k_img.detach()) 
        
        if self.args.alpha_afd_loss > 0.:
            img_fusion_feat = torch.cat([normalized_all_image_features, t_all_image_features], dim=1)
            txt_fusion_feat = torch.cat([normalized_all_text_features, t_all_text_features], dim=1)
            img_fusion_feat = self.visual_fusion_proj(img_fusion_feat)
            txt_fusion_feat = self.text_fusion_proj(txt_fusion_feat)
            img_fusion_feat = F.normalize(img_fusion_feat, dim=1)
            txt_fusion_feat = F.normalize(txt_fusion_feat, dim=1)
            
            logits_per_fusion_image = self.fusion_logit_scale * img_fusion_feat @ txt_fusion_feat.T
            logits_per_fusion_text = logits_per_fusion_image.T
            afd_loss = (
                F.cross_entropy(logits_per_fusion_image, labels) +
                F.cross_entropy(logits_per_fusion_text, labels)
            ) / 2
            
            
        ckd_loss = self.args.alpha_ckd_loss * ckd_loss
        icl_loss = self.args.alpha_icl_loss * icl_loss
        cross_kd_loss = self.args.alpha_cross_kd_loss * cross_kd_loss
        fd_loss = self.args.alpha_fd_loss * fd_loss
        gd_loss = self.args.alpha_gd_loss * gd_loss
        afd_loss = self.args.alpha_afd_loss * afd_loss
        
        return task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss
    
class MyOrthogonal(nn.Module):
    def __init__(self, ds, dt):
        super(MyOrthogonal, self).__init__()
        self.ds = ds
        self.dt = dt
        self.weight = nn.Parameter(torch.empty((dt, dt)))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Enforce skew-symmetry
        W = (self.weight - self.weight.T) / 2
        A = torch.linalg.matrix_exp(W)
        P = A[:, 0:self.ds]
        y = F.linear(x, P)
        return y
    
class MultiClipLoss(nn.Module):
    def __init__(
            self,
            args,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.args = args
        assert isinstance(args.t_embed_dim,list)
        self.visual_proj = nn.ModuleList()
        self.text_proj = nn.ModuleList()
        for t_embed_dim in args.t_embed_dim:
            # self.visual_proj.append(torch.nn.utils.parametrizations.orthogonal(nn.Linear(args.s_embed_dim, t_embed_dim, bias=False)))
            # self.text_proj.append(torch.nn.utils.parametrizations.orthogonal(nn.Linear(args.s_embed_dim, t_embed_dim, bias=False)))
            self.visual_proj.append(MyOrthogonal(args.s_embed_dim, t_embed_dim))
            self.text_proj.append(MyOrthogonal(args.s_embed_dim, t_embed_dim))

        if args.alpha_afd_loss > 0.:
            self.visual_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
            self.text_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
            
        # cache state
        self.prev_num_logits = 0
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.ParameterList([nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) for _ in args.t_embed_dim])
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, \
        t_image_features, t_text_features, t_logit_scale):
        """
        t_image_features,t_text_features,t_logit_scale:list of each teacher's output
        """
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            all_teacher_logits = []
            all_teacher_all_image_features = []
            all_teacher_all_text_features = []

            for t_image_f,t_text_f,t_logit_sc in zip(t_image_features,t_text_features,t_logit_scale):

                t_all_image_features, t_all_text_features = gather_features(
                    t_image_f, t_text_f,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                
                t_logits_per_image = t_logit_sc * t_all_image_features @ t_all_text_features.T
                t_logits_per_text = t_logits_per_image.T
                dic = {
                    't_logits_per_image':t_logits_per_image,
                    't_logits_per_text':t_logits_per_text,
                }
                all_teacher_logits.append(dic)
                all_teacher_all_image_features.append(t_all_image_features)
                all_teacher_all_text_features.append(t_all_text_features)

            normalized_image_features = F.normalize(image_features, dim=-1)
            normalized_text_features = F.normalize(text_features, dim=-1)
            normalized_all_image_features = F.normalize(all_image_features, dim=-1)
            normalized_all_text_features = F.normalize(all_text_features, dim=-1)
            
            if self.local_loss:
                logits_per_image = logit_scale * normalized_image_features @ normalized_all_text_features.T
                logits_per_text = logit_scale * normalized_text_features @ normalized_all_image_features.T
            else:
                logits_per_image = logit_scale * normalized_all_image_features @ normalized_all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            all_teacher_logits = []
            all_teacher_all_image_features = []
            all_teacher_all_text_features = []

            all_image_features = image_features
            all_text_features = text_features
            for t_image_f,t_text_f,t_logit_sc in zip(t_image_features,t_text_features,t_logit_scale):
                t_all_image_features = t_image_f
                t_all_text_features = t_text_f
                
                t_logits_per_image = t_logit_sc * t_all_image_features @ t_all_text_features.T
                t_logits_per_text = t_logits_per_image.T
                dic = {
                    't_logits_per_image':t_logits_per_image,
                    't_logits_per_text':t_logits_per_text,
                }
                all_teacher_logits.append(dic)
                all_teacher_all_image_features.append(t_all_image_features)
                all_teacher_all_text_features.append(t_all_text_features)

            normalized_image_features = F.normalize(image_features,dim=-1)
            normalized_text_features = F.normalize(text_features,dim=-1)
            normalized_all_image_features = normalized_image_features
            normalized_all_text_features = normalized_text_features
            logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.T
            logits_per_text = logit_scale * normalized_text_features @ normalized_image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        all_image_features_proj = []
        all_text_features_proj = []
        for visual_proj, text_proj in zip(self.visual_proj,self.text_proj):
            all_image_features_proj.append(visual_proj(all_image_features))
            all_text_features_proj.append(text_proj(all_text_features))
        

        normalized_all_image_features = [F.normalize(i, dim=-1) for i in all_image_features_proj]
        normalized_all_text_features = [F.normalize(t, dim=-1) for t in all_text_features_proj]#list
        
        fd_loss = torch.stack([F.mse_loss(s_all_image_features,t_all_image_features) + F.mse_loss(s_all_text_features,t_all_text_features) for s_all_image_features,t_all_image_features,s_all_text_features,t_all_text_features in zip(normalized_all_image_features, all_teacher_all_image_features,\
                                                                                                                                                                                                                            normalized_all_text_features, all_teacher_all_text_features)]).mean()
        
        
            
        logits_per_s_image_to_t_text = [cross_logit_scale * s_i @ t_t.T for cross_logit_scale, s_i, t_t in zip(self.cross_logit_scale, normalized_all_image_features, all_teacher_all_text_features)]
        logits_per_s_text_to_t_image = [cross_logit_scale * s_t @ t_i.T for cross_logit_scale, s_t, t_i in zip(self.cross_logit_scale, normalized_all_text_features, all_teacher_all_image_features)]
        
        task_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        ckd_loss = torch.tensor(0.).cuda() 
        icl_loss = torch.tensor(0.).cuda() 
        cross_kd_loss = torch.tensor(0.).cuda() 
        gd_loss = torch.tensor(0.).cuda() 
        afd_loss = torch.tensor(0.).cuda() 
        
        if self.args.alpha_icl_loss > 0:
            icl_loss = (
                F.cross_entropy(torch.mean(torch.stack(logits_per_s_image_to_t_text),dim=0), labels) +
                F.cross_entropy(torch.mean(torch.stack(logits_per_s_text_to_t_image),dim=0), labels)
                ) / 2
        #compute mean KL divergence on each student-teacher pair
        if self.args.alpha_ckd_loss > 0 or self.args.alpha_cross_kd_loss > 0:
            all_ckd_loss = []
            all_cross_kd_loss = []
            for dic,logits_per_s_image_to_t_text,logits_per_s_text_to_t_image in zip(all_teacher_logits,logits_per_s_image_to_t_text,logits_per_s_text_to_t_image):
                t_logits_per_image = dic['t_logits_per_image']
                t_logits_per_text = dic['t_logits_per_text']

                ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
                    self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2
                all_ckd_loss.append(ckd_loss)

                cross_kd_loss = (self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
                    self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2
                all_cross_kd_loss.append(cross_kd_loss)

            ckd_loss = torch.mean(torch.stack(all_ckd_loss))
            cross_kd_loss = torch.mean(torch.stack(all_cross_kd_loss))



        if self.args.alpha_gd_loss > 0.:
            with torch.no_grad():
                t_grad_p_img, t_grad_k_txt = get_grad(t_all_image_features, t_all_text_features, t_logit_scale, labels)
                t_grad_p_txt, t_grad_k_img = get_grad(t_all_text_features, t_all_image_features, t_logit_scale, labels)
            
            s_grad_p_img, s_grad_k_txt = get_grad(normalized_all_image_features, normalized_all_text_features, logit_scale, labels)
            s_grad_p_txt, s_grad_k_img = get_grad(normalized_all_text_features, normalized_all_image_features, logit_scale, labels)

            gd_loss = F.mse_loss(s_grad_p_img, t_grad_p_img.detach()) +\
                F.mse_loss(s_grad_k_txt, t_grad_k_txt.detach()) +\
                    F.mse_loss(s_grad_p_txt, t_grad_p_txt.detach()) +\
                        F.mse_loss(s_grad_k_img, t_grad_k_img.detach()) 
        
        if self.args.alpha_afd_loss > 0.:
            img_fusion_feat = torch.cat([normalized_all_image_features, t_all_image_features], dim=1)
            txt_fusion_feat = torch.cat([normalized_all_text_features, t_all_text_features], dim=1)
            img_fusion_feat = self.visual_fusion_proj(img_fusion_feat)
            txt_fusion_feat = self.text_fusion_proj(txt_fusion_feat)
            img_fusion_feat = F.normalize(img_fusion_feat, dim=1)
            txt_fusion_feat = F.normalize(txt_fusion_feat, dim=1)
            
            logits_per_fusion_image = self.fusion_logit_scale * img_fusion_feat @ txt_fusion_feat.T
            logits_per_fusion_text = logits_per_fusion_image.T
            afd_loss = (
                F.cross_entropy(logits_per_fusion_image, labels) +
                F.cross_entropy(logits_per_fusion_text, labels)
            ) / 2
            
            
        ckd_loss = self.args.alpha_ckd_loss * ckd_loss
        icl_loss = self.args.alpha_icl_loss * icl_loss
        cross_kd_loss = self.args.alpha_cross_kd_loss * cross_kd_loss
        fd_loss = self.args.alpha_fd_loss * fd_loss
        gd_loss = self.args.alpha_gd_loss * gd_loss
        afd_loss = self.args.alpha_afd_loss * afd_loss
        
        return task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss
    

class DRKDClipLoss(nn.Module):

    def __init__(
            self,
            args,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.args = args


        self.visual_proj = nn.ModuleList([MyOrthogonal(args.s_embed_dim, args.t_embed_dim) for _ in range(2)])
        self.text_proj = nn.ModuleList([MyOrthogonal(args.s_embed_dim, args.t_embed_dim) for _ in range(2)])
            
        # cache state
        self.prev_num_logits = 0
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, \
        t_image_features, t_text_features, t_logit_scale):
        device = image_features.device
        t_text_features = t_text_features.contiguous()
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            t_all_image_features, t_all_text_features = gather_features(
                t_image_features, t_text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            decompose_t_all_image_features = list(t_all_image_features.chunk(2,dim=-1))
            decompose_t_all_image_features[0] = F.normalize(decompose_t_all_image_features[0],dim=-1)
            decompose_t_all_image_features[1] = F.normalize(decompose_t_all_image_features[1],dim=-1)

            decompose_t_all_text_features = list(t_all_text_features.chunk(2,dim=-1))
            decompose_t_all_text_features[0] = F.normalize(decompose_t_all_text_features[0],dim=-1)
            decompose_t_all_text_features[1] = F.normalize(decompose_t_all_text_features[1],dim=-1)

            t_logits_per_image_list = []
            t_logits_per_text_list = []
            for i in range(2):
                t_logits_per_image = t_logit_scale * decompose_t_all_image_features[i] @ decompose_t_all_text_features[i].T
                t_logits_per_text = t_logits_per_image.T
                t_logits_per_image_list.append(t_logits_per_image)
                t_logits_per_text_list.append(t_logits_per_text)

            normalized_image_features = F.normalize(image_features, dim=-1)
            normalized_text_features = F.normalize(text_features, dim=-1)
            normalized_all_image_features = F.normalize(all_image_features, dim=-1)
            normalized_all_text_features = F.normalize(all_text_features, dim=-1)
            
            if self.local_loss:
                logits_per_image = logit_scale * normalized_image_features @ normalized_all_text_features.T
                logits_per_text = logit_scale * normalized_text_features @ normalized_all_image_features.T
            else:
                logits_per_image = logit_scale * normalized_all_image_features @ normalized_all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            all_image_features = image_features
            all_text_features = text_features
            t_all_image_features = t_image_features
            t_all_text_features = t_text_features

            decompose_t_all_image_features = list(t_all_image_features.chunk(2,dim=-1))
            decompose_t_all_image_features[0] = F.normalize(decompose_t_all_image_features[0],dim=-1)
            decompose_t_all_image_features[1] = F.normalize(decompose_t_all_image_features[1],dim=-1)

            decompose_t_all_text_features = list(t_all_text_features.chunk(2,dim=-1))
            decompose_t_all_text_features[0] = F.normalize(decompose_t_all_text_features[0],dim=-1)
            decompose_t_all_text_features[1] = F.normalize(decompose_t_all_text_features[1],dim=-1)

            t_logits_per_image_list = []
            t_logits_per_text_list = []
            for i in range(2):
                t_logits_per_image = t_logit_scale * decompose_t_all_image_features[i] @ decompose_t_all_text_features[i].T
                t_logits_per_text = t_logits_per_image.T
                t_logits_per_image_list.append(t_logits_per_image)
                t_logits_per_text_list.append(t_logits_per_text)

            normalized_image_features = F.normalize(image_features,dim=-1)
            normalized_text_features = F.normalize(text_features,dim=-1)
            logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.T
            logits_per_text = logit_scale * normalized_text_features @ normalized_image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        proj_image_features = []
        proj_text_features = []
        for i in range(2):
            proj_image_features.append(F.normalize(self.visual_proj[i](all_image_features),dim=-1))
            proj_text_features.append(F.normalize(self.text_proj[i](all_text_features),dim=-1))
            

        fd_loss = sum([(F.mse_loss(s_i, t_i) +\
            F.mse_loss(s_t, t_t))/2 for \
                s_i,\
                t_i,\
                s_t,\
                t_t \
                    in zip(proj_image_features, \
                            decompose_t_all_image_features, \
                            proj_text_features, \
                            decompose_t_all_text_features)])/2
                                                                                                                                                                                                                                            
            
        logits_per_s_image_to_t_text_list = [self.cross_logit_scale * s_i @ t_t.T for s_i, t_t in zip(proj_image_features, decompose_t_all_text_features)]
        logits_per_s_text_to_t_image_list = [self.cross_logit_scale * s_t @ t_i.T for s_t, t_i in zip(proj_text_features, decompose_t_all_image_features)]
        
        task_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        ckd_loss = torch.tensor(0.).cuda() 
        icl_loss = torch.tensor(0.).cuda() 
        cross_kd_loss = torch.tensor(0.).cuda() 
        
        icl_loss = sum([(
            F.cross_entropy(logits_per_s_image_to_t_text, labels) +
            F.cross_entropy(logits_per_s_text_to_t_image, labels)
            ) / 2 for logits_per_s_image_to_t_text,logits_per_s_text_to_t_image in zip(logits_per_s_image_to_t_text_list, logits_per_s_text_to_t_image_list)])/2
        
        ckd_loss = sum([(self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2 for t_logits_per_image, t_logits_per_text in zip(t_logits_per_image_list, t_logits_per_text_list)])/2
        
        cross_kd_loss = sum([(self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2 for logits_per_s_image_to_t_text,t_logits_per_image,logits_per_s_text_to_t_image,t_logits_per_text  in zip(logits_per_s_image_to_t_text_list,t_logits_per_image_list, logits_per_s_text_to_t_image_list,t_logits_per_text_list)])/2
        #kd_loss = (F.cross_entropy(logits_per_image, F.softmax(, dim=1)) \
        #    + F.cross_entropy(logits_per_text, F.softmax(t_logits_per_text.detach(), dim=1))) / 2

        ckd_loss = self.args.alpha_ckd_loss * ckd_loss
        icl_loss = self.args.alpha_icl_loss * icl_loss
        cross_kd_loss = self.args.alpha_cross_kd_loss * cross_kd_loss
        fd_loss = self.args.alpha_fd_loss * fd_loss
        
        return task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss


class MultiDRKDClipLoss(nn.Module):
    def __init__(
            self,
            args,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.args = args


        self.visual_proj = nn.ModuleList([MyOrthogonal(args.s_embed_dim, args.t_embed_dim) for _ in range(2)])
        self.text_proj = nn.ModuleList([MyOrthogonal(args.s_embed_dim, args.t_embed_dim) for _ in range(2)])
            
        # cache state
        self.prev_num_logits = 0
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, \
        t_image_features, t_text_features, t_logit_scale):
        """"
        image_features:[B,D]
        text_features:[3,B,D]
        t_image_features:[B,2D]
        t_text_features:[3,B,2D]
        """
        device = image_features.device
        text_features = list(text_features.chunk(3,dim=0))
        t_text_features = list(t_text_features.chunk(3,dim=0))
        for i in range(len(t_text_features)):
            t_text_features[i] = t_text_features[i].contiguous().squeeze()
            text_features[i] = text_features[i].contiguous().squeeze()

        if self.world_size > 1:
            all_text_features_list = []
            t_all_text_features_list = []
            all_image_features = gather_image_features(
                image_features=image_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod
            )
            t_all_image_features = gather_image_features(
                image_features=t_image_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod
            )
            t_all_image_features = list(t_all_image_features.chunk(2,dim=-1))#[B,768]x2
            t_all_image_features = [F.normalize(x,dim=-1) for x in t_all_image_features]

            for i in range(3):
                all_text_features_list.append(gather_text_features(
                    text_features=text_features[i],
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod
                ))#[B,D]x3
                t_all_t_f = gather_text_features(
                    text_features=t_text_features[i],
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod
                )#[B,768*2]
                t_all_t_f = list(t_all_t_f.chunk(2,dim=-1))#[B,768]x2
                t_all_t_f = [F.normalize(x,dim=-1) for x in t_all_t_f]
                t_all_text_features_list.append(t_all_t_f)#[[B,768]x2]x3
            normalized_all_text_features_list = [F.normalize(all_text_features,dim=-1) for all_text_features in all_text_features_list]#[B,D]x3

            t_logits_per_image_list = []
            t_logits_per_text_list = []
            for i in range(3):
                t_pi = []
                t_pt = []
                for j in range(2):
                    t_logits_per_image = t_logit_scale * t_all_image_features[j] @ t_all_text_features_list[i][j].T
                    t_logits_per_text = t_logits_per_image.T
                    t_pi.append(t_logits_per_image)
                    t_pt.append(t_logits_per_text)
                t_logits_per_image_list.append(t_pi)
                t_logits_per_text_list.append(t_pt)
            t_logits_per_image_list = [item for sublist in t_logits_per_image_list for item in sublist]
            t_logits_per_text_list = [item for sublist in t_logits_per_text_list for item in sublist]

            normalized_image_features = F.normalize(image_features, dim=-1)

            normalized_all_image_features = F.normalize(all_image_features, dim=-1)
            
            if self.local_loss:
                logits_per_image = [logit_scale * normalized_image_features @ f.T for f in normalized_all_text_features_list]
                logits_per_text = [logit_scale * F.normalize(f,dim=-1) @ normalized_all_image_features.T for f in text_features]
            else:
                logits_per_image = [logit_scale * normalized_all_image_features @ f.T for f in normalized_all_text_features_list]
                logits_per_text = [x.T for x in logits_per_image]
            logits_per_image = torch.mean(torch.stack(logits_per_image),dim=0)
            logits_per_text = torch.mean(torch.stack(logits_per_text),dim=0)
        else:
            raise ValueError("not implemented single gpu")

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        proj_image_features = []
        proj_text_features = []
        for i in range(2):
            proj_image_features.append(F.normalize(self.visual_proj[i](all_image_features),dim=-1))
        for i in range(3):
            proj_one_text_features = []
            for j in range(2):
                proj_one_text_features.append(F.normalize(self.text_proj[j](all_text_features_list[i]),dim=-1))#同一个文本，两个投影空间
            proj_text_features.append(proj_one_text_features)#[[B,768]x2]x3

        fd_loss_i = sum([F.mse_loss(s_i,t_i) for s_i,t_i in zip(proj_image_features, t_all_image_features)])/len(proj_image_features)
        fd_loss_t = torch.tensor(0.).cuda()
        for i in range(3):
            for j in range(2):
                fd_loss_t = fd_loss_t + F.mse_loss(proj_text_features[i][j],t_all_text_features_list[i][j])
        fd_loss_t = fd_loss_t/6
        fd_loss = fd_loss_t + fd_loss_i
                                                                                                                                                                                                                                        
        logits_per_s_image_to_t_text_list = [[self.cross_logit_scale * s_i @ t_t.T for s_i, t_t in zip(proj_image_features, t_all_text_features_list[k])] for k in range(3)]
        logits_per_s_text_to_t_image_list = [[self.cross_logit_scale * s_t @ t_i.T for s_t, t_i in zip(proj_text_features[k], t_all_image_features)] for k in range(3)]
        # 对于图像到文本的 logits
        flat_logits_per_s_image_to_t_text = [item for sublist in logits_per_s_image_to_t_text_list for item in sublist]

        # 对于文本到图像的 logits
        flat_logits_per_s_text_to_t_image = [item for sublist in logits_per_s_text_to_t_image_list for item in sublist]

        
        task_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        ckd_loss = torch.tensor(0.).cuda() 
        icl_loss = torch.tensor(0.).cuda() 
        cross_kd_loss = torch.tensor(0.).cuda() 
        
        icl_loss = sum([(
            F.cross_entropy(logits_per_s_image_to_t_text, labels) +
            F.cross_entropy(logits_per_s_text_to_t_image, labels)
            ) / 2 for logits_per_s_image_to_t_text,logits_per_s_text_to_t_image in zip(flat_logits_per_s_image_to_t_text, flat_logits_per_s_text_to_t_image)])
        
        ckd_loss = sum([(self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2 for t_logits_per_image, t_logits_per_text in zip(t_logits_per_image_list, t_logits_per_text_list)])/2
        
        cross_kd_loss = sum([(self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2 for logits_per_s_image_to_t_text,t_logits_per_image,logits_per_s_text_to_t_image,t_logits_per_text in zip(flat_logits_per_s_image_to_t_text,t_logits_per_image_list, flat_logits_per_s_text_to_t_image,t_logits_per_text_list)])/2
        #kd_loss = (F.cross_entropy(logits_per_image, F.softmax(, dim=1)) \
        #    + F.cross_entropy(logits_per_text, F.softmax(t_logits_per_text.detach(), dim=1))) / 2

        ckd_loss = self.args.alpha_ckd_loss * ckd_loss
        icl_loss = self.args.alpha_icl_loss * icl_loss
        cross_kd_loss = self.args.alpha_cross_kd_loss * cross_kd_loss
        fd_loss = self.args.alpha_fd_loss * fd_loss
        
        return task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss