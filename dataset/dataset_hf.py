# data_hf.py

import os
import torch
from torch.utils import data
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate
from os.path import join as pjoin

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# ==========================================
# 1. VQ-VAE 训练用 Dataset (motion only)
# ==========================================
class HF_VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4, cache_dir=None):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        print(f"Loading {dataset_name} Train dataset from HuggingFace...")
        self.dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)

        num_cores = min(8, os.cpu_count() or 1)
        self.dataset = self.dataset.filter(
            lambda meta: meta['num_frames'] >= self.window_size,
            input_columns=['meta_data'],
            num_proc=num_cores
        )

        self.dataset = self.dataset.with_format("numpy")
        print(f"HF VQ Train Dataset Loaded! Total valid motions: {len(self.dataset)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_item = self.dataset[item]
        motion = data_item['motion']

        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx: idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return motion


# ==========================================
# 2. VQ-VAE 评测用 Dataset (motion + text)
# ==========================================
class HF_Text2MotionDataset(data.Dataset):
    """用于 VQ-VAE 和 Transformer 的 evaluation"""
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4, cache_dir=None):
        self.max_length = 20
        self.pointer = 0
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.dataset_name = dataset_name
        self.is_test = is_test

        if dataset_name == 't2m':
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            min_motion_len = 40
            fps = 20
        elif dataset_name == 'kit':
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            min_motion_len = 24
            fps = 12.5

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        split_name = "test" if is_test else "val"
        print(f"Loading {dataset_name} {split_name} dataset from HuggingFace...")

        hf_dataset = load_dataset("TeoGchx/HumanML3D", split=split_name, cache_dir=cache_dir)

        data_dict = {}
        new_name_list = []
        length_list = []

        for i in tqdm(range(len(hf_dataset)), desc=f"Building eval data_dict ({split_name})"):
            try:
                data_item = hf_dataset[i]
                motion = np.array(data_item['motion'], dtype=np.float32)
                name = data_item['meta_data']['name']

                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                raw_text_str = data_item['caption']
                text_lines = [line for line in raw_text_str.split('\n') if line.strip() != '']

                text_data = []
                flag = False

                for line in text_lines:
                    text_dict = {}
                    line_split = line.strip().split('#')
                    if len(line_split) < 4:
                        continue

                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        try:
                            n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                            if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {
                                'motion': n_motion,
                                'length': len(n_motion),
                                'text': [text_dict]
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                        except:
                            pass

                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))

            except Exception as e:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        print(f"HF Eval Dataset Loaded! Total valid entries: {len(self.data_dict)}, after pointer: {len(self)}")

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def forward_transform(self, data_in):
        return (data_in - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]

        motion, m_length, text_list = data['motion'], data['length'], data['text']

        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])

        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]

        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name


# ==========================================
# 3. Tokenize 用 Dataset (编码 motion → token)
#    对应原始 dataset_tokenize.py
# ==========================================
class HF_TokenizeDataset(data.Dataset):
    """
    用于将 HF 训练集的 motion 通过 VQ-VAE encoder 编码为离散 token。
    与原始 VQMotionDataset 保持一致的处理逻辑：
    - 过滤过短/过长的 motion
    - 长度按 unit_length 对齐
    - 随机裁剪起始位置
    - Z-normalization
    每条样本返回: (motion_tensor, name_str)
    """
    def __init__(self, dataset_name, unit_length=4, cache_dir=None):
        self.dataset_name = dataset_name
        self.unit_length = unit_length

        if dataset_name == 't2m':
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            min_motion_len = 40
        elif dataset_name == 'kit':
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            min_motion_len = 24

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        print(f"Loading {dataset_name} Train dataset from HuggingFace for tokenization...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)

        # 与原始 VQMotionDataset 一致：预加载到内存并过滤
        self.data_dict = {}
        self.name_list = []
        self.length_list = []

        for i in tqdm(range(len(hf_dataset)), desc="Loading tokenize data"):
            try:
                data_item = hf_dataset[i]
                motion = np.array(data_item['motion'], dtype=np.float32)
                name = data_item['meta_data']['name']

                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                self.data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'name': name
                }
                self.name_list.append(name)
                self.length_list.append(len(motion))
            except:
                pass

        self.length_arr = np.array(self.length_list)
        print(f"HF Tokenize Dataset Loaded! Total valid motions: {len(self.data_dict)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        # 与原始 VQMotionDataset 一致：长度按 unit_length 对齐
        m_length = (m_length // self.unit_length) * self.unit_length

        # 随机裁剪起始位置
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]

        # Z-normalization
        motion = (motion - self.mean) / self.std
        return motion, name


def tokenize_collate_fn(batch):
    """
    TokenizeDataset 每条 motion 长度不同，batch_size=1 时无需特殊处理。
    batch_size>1 时需要 padding，这里简单支持 batch_size=1。
    """
    # batch is a list of (motion_np, name_str)
    # 只支持 batch_size=1
    motions, names = zip(*batch)
    # motions[0] shape: (seq_len, feat_dim)
    motion_tensor = torch.from_numpy(np.array(motions))  # (1, seq_len, feat_dim)
    return motion_tensor, names


# ==========================================
# 4. Transformer 训练用 Dataset (text + motion tokens)
#    对应原始 dataset_TM_train.py 的 Text2MotionDataset
# ==========================================
class HF_Text2MotionTokenDataset(data.Dataset):
    def __init__(self, dataset_name, codebook_size=1024, tokenizer_name=None,
                 unit_length=4, cache_dir=None, vq_dir=None):
        self.max_length = 64
        self.dataset_name = dataset_name
        self.unit_length = unit_length

        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        if dataset_name == 't2m':
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
        elif dataset_name == 'kit':
            fps = 12.5
            self.max_motion_length = 26 if unit_length == 8 else 51
        self.fps = fps

        # vq_dir 设置
        if vq_dir is not None:
            self.vq_dir = vq_dir
        else:
            data_root = "./dataset/KIT-ML" if dataset_name == 'kit' else "./dataset/HumanML3D"
            self.vq_dir = pjoin(data_root, tokenizer_name) if tokenizer_name else pjoin(data_root, 'vq_tokens')

        print(f"Loading {dataset_name} Train dataset from HuggingFace...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)
        
        # ===== 优化1: 预先获取所有可用的 vq token 文件 =====
        print("Scanning available VQ token files...")
        available_tokens = set(
            f[:-4] for f in os.listdir(self.vq_dir) if f.endswith('.npy')
        )
        print(f"Found {len(available_tokens)} VQ token files")

        # ===== 优化2: 批量提取需要的字段，避免逐个访问 =====
        print("Extracting data from HF dataset...")
        # 一次性提取所有需要的数据
        all_names = [item['name'] for item in hf_dataset['meta_data']]
        all_captions = hf_dataset['caption']
        
        # ===== 优化3: 使用字典推导 + 并行预加载 token =====
        data_dict = {}
        new_name_list = []
        
        # 预加载所有需要的 token（可选：用多进程加速）
        valid_indices = [i for i, name in enumerate(all_names) if name in available_tokens]
        print(f"Valid samples with VQ tokens: {len(valid_indices)}")
        
        for i in tqdm(valid_indices, desc="Building data_dict(train)"):
            name = all_names[i]
            raw_text_str = all_captions[i]
            
            # 解析 caption（简化逻辑）
            text_data, sub_motions = self._parse_captions(raw_text_str, name, fps, unit_length)
            
            if not text_data and not sub_motions:
                continue
                
            # 延迟加载 token（在 __getitem__ 中加载）
            if text_data:
                data_dict[name] = {'text': text_data, 'token_file': pjoin(self.vq_dir, f'{name}.npy')}
                new_name_list.append(name)
            
            for sub_name, sub_text, f_tag, to_tag in sub_motions:
                data_dict[sub_name] = {
                    'text': [sub_text],
                    'token_file': pjoin(self.vq_dir, f'{name}.npy'),
                    'slice': (int(f_tag * fps / unit_length), int(to_tag * fps / unit_length))
                }
                new_name_list.append(sub_name)

        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"Dataset Loaded! Total entries: {len(self.data_dict)}")

    def _parse_captions(self, raw_text_str, name, fps, unit_length):
        """解析 caption 字符串，返回完整动作文本和子动作列表"""
        text_data = []
        sub_motions = []
        
        for line in raw_text_str.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('#')
            if len(parts) < 4:
                continue
            
            try:
                caption = parts[0]
                t_tokens = parts[1].split(' ')
                f_tag = float(parts[2]) if parts[2] else 0.0
                to_tag = float(parts[3]) if parts[3] else 0.0
                
                if np.isnan(f_tag): f_tag = 0.0
                if np.isnan(to_tag): to_tag = 0.0
                
                text_dict = {'caption': caption, 'tokens': t_tokens}
                
                if f_tag == 0.0 and to_tag == 0.0:
                    text_data.append(text_dict)
                else:
                    start_idx = int(f_tag * fps / unit_length)
                    end_idx = int(to_tag * fps / unit_length)
                    if start_idx < end_idx:
                        sub_name = f'{name}_{f_tag}_{to_tag}'
                        sub_motions.append((sub_name, text_dict, f_tag, to_tag))
            except:
                continue
        
        return text_data, sub_motions

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        text_data = random.choice(data['text'])
        caption = text_data['caption']
        
        # 延迟加载 token
        m_token_list = np.load(data['token_file'])
        
        # --- 修复逻辑：确保 m_tokens 是一维的 ---
        # 如果 m_token_list 是二维 (N, L)，random.choice 会选出其中一行 (L,)
        # 如果已经是 (L,)，则直接使用
        m_tokens = random.choice(m_token_list) if m_token_list.ndim > 1 else m_token_list
        
        # 如果是子动作，对这个 1D 序列进行切片
        if 'slice' in data:
            start, end = data['slice']
            m_tokens = m_tokens[start:end]
        # --------------------------------------

        # 随机 drop token (数据增强)
        if np.random.random() < 1/3:
            if len(m_tokens) > 1: # 增加校验防止切空
                m_tokens = m_tokens[:-1] if np.random.random() < 0.5 else m_tokens[1:]
        
        m_tokens_len = len(m_tokens)

        # Padding 逻辑 (参考 dataset_TM_train.py 确保拼接维度一致)
        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate([
                m_tokens, 
                np.array([self.mot_end_idx], dtype=m_tokens.dtype), 
                np.full((self.max_motion_length - 1 - m_tokens_len), self.mot_pad_idx, dtype=m_tokens.dtype)
            ], axis=0)
        else:
            m_tokens = np.concatenate([
                m_tokens, 
                np.array([self.mot_end_idx], dtype=m_tokens.dtype)
            ], axis=0)

        # 最终 reshape(-1) 确保返回形状为 (max_motion_length,)
        return caption, m_tokens.reshape(-1).astype(np.int64), m_tokens_len


# ==========================================
# 统一的 DataLoader 获取接口
# ==========================================

# --- VQ-VAE 训练 ---
def get_train_loader(dataset_name, batch_size, window_size=64, unit_length=4, num_workers=8, cache_dir=None):
    train_set = HF_VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, cache_dir=cache_dir)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return train_loader

# --- VQ-VAE / Transformer 评测 ---
def get_val_loader(dataset_name, batch_size, w_vectorizer, is_test=False, unit_length=4, num_workers=8, cache_dir=None):
    val_set = HF_Text2MotionDataset(dataset_name, is_test=is_test, w_vectorizer=w_vectorizer, unit_length=unit_length, cache_dir=cache_dir)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return val_loader

# --- Tokenize (编码 motion → VQ token) ---
def get_tokenize_loader(dataset_name, batch_size=1, unit_length=4, num_workers=0, cache_dir=None):
    """batch_size 建议为 1，因为不同 motion 长度不同"""
    token_set = HF_TokenizeDataset(dataset_name, unit_length=unit_length, cache_dir=cache_dir)
    token_loader = torch.utils.data.DataLoader(
        token_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tokenize_collate_fn,
        drop_last=False
    )
    return token_loader

# --- Transformer 训练 ---
def get_trans_train_loader(dataset_name, batch_size, codebook_size, tokenizer_name=None,
                           unit_length=4, num_workers=8, cache_dir=None, vq_dir=None):
    train_set = HF_Text2MotionTokenDataset(
        dataset_name, codebook_size=codebook_size,
        tokenizer_name=tokenizer_name, unit_length=unit_length,
        cache_dir=cache_dir, vq_dir=vq_dir
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return train_loader