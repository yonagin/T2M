# train_t2m_trans.py

import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
from dataset import dataset_hf
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

# vq_dir: 存放 tokenized motion 的目录
# 当使用 HF 模式时，vq_dir 放在 out_dir 下而非 dataset 目录下
if hasattr(args, 'cache_dir') and args.cache_dir is not None:
    args.vq_dir = os.path.join(args.out_dir, f'{args.vq_name}')
else:
    args.vq_dir = os.path.join(
        "./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D",
        f'{args.vq_name}'
    )
os.makedirs(args.vq_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Eval Dataloader ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
clip.model.convert_weights(clip_model)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate
)

trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=args.embed_dim_gpt,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate
)

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu', weights_only=False)
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu', weights_only=False)
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

##### ---- Tokenize & Build DataLoaders ---- #####
use_hf = hasattr(args, 'cache_dir') and args.cache_dir is not None

if use_hf:
    logger.info("Using HuggingFace dataset mode")

    # --- Step 1: Tokenize ---
    # 检查是否已经完成 tokenization（通过检查 vq_dir 是否有文件）
    existing_tokens = [f for f in os.listdir(args.vq_dir) if f.endswith('.npy')]
    if len(existing_tokens) == 0:
        logger.info("No existing tokens found, starting tokenization...")
        train_loader_token = dataset_hf.get_tokenize_loader(
            args.dataname, batch_size=1, unit_length=2 ** args.down_t,
            num_workers=0, cache_dir=args.cache_dir
        )
        with torch.no_grad():
            for batch in train_loader_token:
                pose, name = batch
                pose = pose.cuda().float()  # (1, seq_len, feat_dim)
                target = net.encode(pose)
                target = target.cpu().numpy()
                np.save(pjoin(args.vq_dir, name[0] + '.npy'), target)
        logger.info(f"Tokenization complete! Saved to {args.vq_dir}")
    else:
        logger.info(f"Found {len(existing_tokens)} existing token files in {args.vq_dir}, skipping tokenization.")

    # --- Step 2: Build train loader ---
    train_loader = dataset_hf.get_trans_train_loader(
        args.dataname, args.batch_size, args.nb_code,
        tokenizer_name=args.vq_name, unit_length=2 ** args.down_t,
        cache_dir=args.cache_dir, vq_dir=args.vq_dir
    )
    train_loader_iter = dataset_hf.cycle(train_loader)

    # --- Step 3: Build val loader ---
    val_loader = dataset_hf.get_val_loader(
        args.dataname, batch_size=32, w_vectorizer=w_vectorizer,
        is_test=False, unit_length=2 ** args.down_t,
        cache_dir=args.cache_dir
    )

else:
    logger.info("Using local filesystem dataset mode")

    # --- Step 1: Tokenize ---
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2 ** args.down_t)
    with torch.no_grad():
        for batch in train_loader_token:
            pose, name = batch
            bs, seq = pose.shape[0], pose.shape[1]
            pose = pose.cuda().float()
            target = net.encode(pose)
            target = target.cpu().numpy()
            np.save(pjoin(args.vq_dir, name[0] + '.npy'), target)

    # --- Step 2: Build train loader ---
    train_loader = dataset_TM_train.DATALoader(
        args.dataname, args.batch_size, args.nb_code,
        args.vq_name, unit_length=2 ** args.down_t
    )
    train_loader_iter = dataset_TM_train.cycle(train_loader)

    # --- Step 3: Build val loader ---
    val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)


##### ---- Training ---- #####
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000.0, 0, 100.0, 0.0, 0.0, 0.0, 100.0    

while nb_iter <= args.total_iter:

    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens  # (bs, max_motion_length)
    target = target.cuda()

    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float()

    input_index = target[:, :-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape, device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape, device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask * input_index + (1 - mask) * r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text)
    cls_pred = cls_pred.contiguous()

    loss_cls = 0.0
    for i in range(bs):
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs

        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)
        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()

    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter == 0:
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0


    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        eval_trans.evaluation_transformer(
            args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
            best_fid, best_iter, best_div, best_top1, best_top2,
            best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper
        )

    if nb_iter == args.total_iter:
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break