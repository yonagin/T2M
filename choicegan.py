import os
import json
import math  # 新增 math 用于计算余弦退火
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import spectral_norm


import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
from dataset import dataset_hf
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer

###############################################################################
# ChoiceGAN VQ-VAE 核心组件
###############################################################################

class CodeDiscriminator(nn.Module):
    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p).squeeze(-1)

def linear_anneal(step: int, total_steps: int, start: float = 1.0, end: float = 0.1) -> float:
    alpha = min(1.0, step / max(total_steps, 1))
    return start * (1.0 - alpha) + end * alpha

def sample_dirichlet_prior(batch_size: int, num_codes: int, alpha: float, device) -> torch.Tensor:
    alpha_tensor = torch.full((num_codes,), alpha, device=device)
    dist = torch.distributions.Dirichlet(alpha_tensor)
    return dist.sample((batch_size,))

def get_soft_p(h: torch.Tensor, codebook: torch.Tensor, temperature: float) -> torch.Tensor:
    h_norm = (h.detach() ** 2).sum(dim=1, keepdim=True)         
    c_norm = (codebook ** 2).sum(dim=1).unsqueeze(0)   
    logits = -(h_norm + c_norm - 2 * h.detach() @ codebook.t()) / max(temperature, 1e-6)
    return F.softmax(logits, dim=-1)                   

def compute_gradient_penalty(discriminator, p_real, p_fake):
    alpha = torch.rand(p_real.size(0), 1, device=p_real.device, dtype=p_real.dtype)
    interpolates = alpha * p_real + (1 - alpha) * p_fake
    interpolates.requires_grad_(True)
    d_interp = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interp)
    gradients = autograd.grad(
        outputs=d_interp, inputs=interpolates, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

# T2M-GPT 原始手动 Warmup (仅给非码本的主网络用)
def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr

# ChoiceGAN 的余弦调度器 Lambda 工厂
def get_cosine_schedule_lambda(warmup_steps, total_steps, min_lr_ratio):
    def scheduler_fn(step):
        if step < warmup_steps:
            return step / max(1.0, warmup_steps)
        progress = (step - warmup_steps) / max(1.0, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + cosine * (1 - min_lr_ratio)
    return scheduler_fn



##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()

torch.manual_seed(args.seed)
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger & Dataset ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)

w_vectorizer = WordVectorizer('./glove', 'our_vab')
if args.dataname == 'kit':
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 21
else:
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

if args.cache_dir is not None:
    train_loader = dataset_hf.get_train_loader(args.dataname, args.batch_size, window_size=args.window_size, unit_length=2**args.down_t, cache_dir=args.cache_dir)
    train_loader_iter = dataset_hf.cycle(train_loader)
    val_loader = dataset_hf.get_val_loader(args.dataname, batch_size=32, w_vectorizer=w_vectorizer, unit_length=2**args.down_t, cache_dir=args.cache_dir)
else:
    train_loader = dataset_VQ.DATALoader(args.dataname, args.batch_size, window_size=args.window_size, unit_length=2**args.down_t)
    train_loader_iter = dataset_VQ.cycle(train_loader)
    val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer, unit_length=2**args.down_t)


##### ---- Network & Discriminator ---- #####
net = vqvae.HumanVQVAE(
    args, args.nb_code, args.code_dim, args.output_emb_width,
    args.down_t, args.stride_t, args.width, args.depth,
    args.dilation_growth_rate, args.vq_act, args.vq_norm)

discriminator = CodeDiscriminator(args.nb_code).cuda()

if args.resume_pth:
    ckpt = torch.load(args.resume_pth, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['net'], strict=True)

net.train()
net.cuda()
discriminator.train()


##### ---- Optimizers & Schedulers (极致切分) ---- #####

# 1. 划分参数组
codebook_params = []
main_params = []
for name, param in net.named_parameters():
    if "quantizer.embedding" in name: 
        codebook_params.append(param)
    else:
        main_params.append(param)

total_iter = args.warm_up_iter + args.total_iter

# ==========================================================
# 2A. 主网络 (非码本参数) - 保留 T2M-GPT 原版策略
# ==========================================================
optimizer_main = optim.AdamW(main_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
# 注意：原版 T2M-GPT 的 scheduler_main 仅在 warm_up 结束后才调用 .step()
scheduler_main = torch.optim.lr_scheduler.MultiStepLR(optimizer_main, milestones=args.lr_scheduler, gamma=args.gamma)

# ==========================================================
# 2B. 码本 (Codebook) - 采用 ChoiceGAN 策略
# ==========================================================
lr_emb = args.lr * args.emb_lr_multiplier
optimizer_codebook = optim.AdamW(codebook_params, lr=lr_emb, betas=(0.9, 0.99), weight_decay=0.0)

min_lr_ratio_cb = args.min_learning_rate / lr_emb
lambda_cb = get_cosine_schedule_lambda(args.warm_up_iter, total_iter, min_lr_ratio_cb)
# 注意：LambdaLR 自带 warmup，所以需要在每一次 iteration 中直接调用 .step()
scheduler_codebook = torch.optim.lr_scheduler.LambdaLR(optimizer_codebook, lr_lambda=lambda_cb)

# ==========================================================
# 2C. 判别器 (Discriminator) - 采用 ChoiceGAN 策略 (余弦退火)
# ==========================================================
optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

min_lr_ratio_d = args.min_learning_rate / args.lr
lambda_d = get_cosine_schedule_lambda(args.warm_up_iter, total_iter, min_lr_ratio_d)
scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lambda_d)


Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

##### ---- Initial evaluation ---- #####
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
    eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0,
                                best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0,
                                best_matching=100, eval_wrapper=eval_wrapper, draw=False)


##### ---- Training Loop ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
avg_d_loss, avg_adv_loss, avg_usage = 0., 0., 0.

stage = 1
usage_ema = None
switch_hits = 0

for nb_iter in range(1, total_iter + 1):
    
    # 仅针对主网络执行 T2M-GPT 原版的手动 warmup
    is_warmup = nb_iter < args.warm_up_iter
    if is_warmup:
        optimizer_main, current_lr = update_lr_warm_up(optimizer_main, nb_iter, args.warm_up_iter, args.lr)
    else:
        current_lr = optimizer_main.param_groups[0]['lr']
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float()

    # ---- 前向传播截取 ----
    x_in = net.vqvae.preprocess(gt_motion)
    h = net.vqvae.encoder(x_in)
    
    x_quantized, loss_commit, perplexity = net.vqvae.quantizer(h)
    x_decoder = net.vqvae.decoder(x_quantized)
    pred_motion = net.vqvae.postprocess(x_decoder)

    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)

    h_flat = net.vqvae.quantizer.preprocess(h)
    codebook = net.vqvae.quantizer.embedding.weight
    temperature = linear_anneal(nb_iter, total_iter, start=args.tau_start, end=args.tau_end)

    # 阶段监控
    code_idx = net.vqvae.quantizer.quantize(h_flat)
    current_usage = (code_idx.unique().numel() / args.nb_code) * 100.0
    if usage_ema is None:
        usage_ema = current_usage
    else:
        usage_ema = 0.9 * usage_ema + 0.1 * current_usage

    if stage == 1 and not is_warmup and usage_ema >= args.switch_usage_threshold:
        switch_hits += 1
        if switch_hits >= 100: 
            stage = 2
            logger.info(f"===> 码本使用率稳定在 {usage_ema:.2f}%. 切换至 Stage 2 (纯 VQ 训练)")
    else:
        switch_hits = 0

    adv_fake_for_g = torch.tensor(0.0, device=gt_motion.device)
    d_loss = torch.tensor(0.0, device=gt_motion.device)

    # ---- 训练判别器 D ----
    if stage == 1:
        p_fake = get_soft_p(h_flat, codebook, temperature)
        p_real = sample_dirichlet_prior(p_fake.size(0), args.nb_code, args.dirichlet_alpha, device=gt_motion.device)
        
        discriminator.requires_grad_(True)
        optimizer_d.zero_grad()

        d_real = discriminator(p_real)
        d_fake = discriminator(p_fake.detach())

        if args.gan_loss_type == "bce":
            d_loss = 0.5 * (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + 
                            F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
        elif args.gan_loss_type == "lsgan":
            d_loss = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + 
                            F.mse_loss(d_fake, torch.zeros_like(d_fake)))
        elif args.gan_loss_type in ("wgan-gp", "wgan_gp"):
            gp_loss = compute_gradient_penalty(discriminator, p_real, p_fake.detach())
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + args.gp_weight * gp_loss

        d_loss.backward()
        optimizer_d.step()

        # 生成器对抗损失
        discriminator.requires_grad_(False)
        logits_adv = discriminator(p_fake)
        if args.gan_loss_type == "bce":
            adv_fake_for_g = F.binary_cross_entropy_with_logits(logits_adv, torch.ones_like(logits_adv))
        elif args.gan_loss_type == "lsgan":
            adv_fake_for_g = F.mse_loss(logits_adv, torch.ones_like(logits_adv))
        elif args.gan_loss_type in ("wgan-gp", "wgan_gp"):  
            adv_fake_for_g = -torch.mean(logits_adv)

    # ---- 训练生成器 G (分为 Main 和 Codebook) ----
    optimizer_main.zero_grad()
    optimizer_codebook.zero_grad()
    
    loss_g = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    if stage == 1:
        loss_g = loss_g + args.lambda_adv * adv_fake_for_g

    loss_g.backward()
        
    optimizer_main.step()
    optimizer_codebook.step()

    # ---- 步进 Schedulers (核心差异) ----
    # 1. T2M-GPT 原始策略：仅在非 warmup 时 MultiStepLR step
    if not is_warmup:
        scheduler_main.step()
        
    # 2. ChoiceGAN 策略：因为 LambdaLR 自带 warmup，所以必须每一轮都 step
    scheduler_codebook.step()
    if stage == 1:
        scheduler_d.step()

    # ---- Logging ----
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_d_loss += d_loss.item()
    avg_adv_loss += adv_fake_for_g.item()
    avg_usage += current_usage

    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_d_loss /= args.print_iter
        avg_adv_loss /= args.print_iter
        avg_usage /= args.print_iter

        if is_warmup:
            logger.info(f"Warmup. Iter {nb_iter} : main_lr {current_lr:.5f} | cb_lr {optimizer_codebook.param_groups[0]['lr']:.5f} \t PPL. {avg_perplexity:.2f} \t Recons. {avg_recons:.5f} \t Usage. {avg_usage:.1f}%")
        else:
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Usage', avg_usage, nb_iter)
            writer.add_scalar('./Train/D_Loss', avg_d_loss, nb_iter)
            writer.add_scalar('./Train/Adv_Loss', avg_adv_loss, nb_iter)
            logger.info(f"Train. Iter {nb_iter} (S{stage}): \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons. {avg_recons:.5f} \t Adv. {avg_adv_loss:.5f} \t Usage. {avg_usage:.1f}%")
            
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        avg_d_loss, avg_adv_loss, avg_usage = 0., 0., 0.

    if nb_iter % args.eval_iter == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
            eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, nb_iter,
                                        best_fid, best_iter, best_div, best_top1, best_top2,
                                        best_top3, best_matching, eval_wrapper=eval_wrapper)

