# data_hf.py (优化版)

import os
import torch
from torch.utils import data
import numpy as np
import random
from datasets import load_dataset, Dataset
from torch.utils.data._utils.collate import default_collate
from os.path import join as pjoin
import functools

NUM_PROC = min(8, os.cpu_count() or 1)


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# ==========================================
# 1. VQ-VAE 训练用 Dataset (motion only) — 保持不变，已经够快
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

        self.dataset = self.dataset.filter(
            lambda meta: meta['num_frames'] >= self.window_size,
            input_columns=['meta_data'],
            num_proc=NUM_PROC
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
# 辅助函数：解析 caption 文本行
# ==========================================
def _parse_text_lines(raw_text_str):
    """解析 caption 字符串为结构化列表"""
    text_lines = [line for line in raw_text_str.split('\n') if line.strip()]
    results = []
    for line in text_lines:
        parts = line.strip().split('#')
        if len(parts) < 4:
            continue
        caption = parts[0]
        tokens = parts[1].split(' ')
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        results.append({
            'caption': caption,
            'tokens': tokens,
            'f_tag': f_tag,
            'to_tag': to_tag,
        })
    return results


# ==========================================
# 2. VQ-VAE 评测用 Dataset (motion + text) — 高效版
# ==========================================
class HF_Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias=5,
                 max_text_len=20, unit_length=4, cache_dir=None):
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

        # ---- 用 map 批量展开，替代逐条 for 循环 ----
        def _expand_to_entries(batch):
            """
            batched map 函数：将每条 HF 样本展开为多条 data entries。
            返回扁平化的列表。
            """
            out_names = []
            out_motions = []
            out_lengths = []
            out_captions_list = []  # list of json-serializable text_data
            out_tokens_list = []

            for idx in range(len(batch['caption'])):
                motion = np.array(batch['motion'][idx], dtype=np.float32)
                name = batch['meta_data'][idx]['name']
                raw_text_str = batch['caption'][idx]

                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                parsed = _parse_text_lines(raw_text_str)
                if not parsed:
                    continue

                whole_text = []
                has_whole = False

                for td in parsed:
                    if td['f_tag'] == 0.0 and td['to_tag'] == 0.0:
                        has_whole = True
                        whole_text.append(td['caption'] + '|||' + ' '.join(td['tokens']))
                    else:
                        f_idx = int(td['f_tag'] * fps)
                        t_idx = int(td['to_tag'] * fps)
                        n_motion = motion[f_idx:t_idx]
                        if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                            continue
                        sub_name = f"SUB_{name}_{td['f_tag']}_{td['to_tag']}"
                        out_names.append(sub_name)
                        out_motions.append(n_motion.tolist())
                        out_lengths.append(len(n_motion))
                        out_captions_list.append(td['caption'])
                        out_tokens_list.append(' '.join(td['tokens']))

                if has_whole:
                    out_names.append(name)
                    out_motions.append(motion.tolist())
                    out_lengths.append(len(motion))
                    # 拼接所有 whole captions
                    out_captions_list.append('@@'.join([t.split('|||')[0] for t in whole_text]))
                    out_tokens_list.append('@@'.join([t.split('|||')[1] for t in whole_text]))

            return {
                'entry_name': out_names,
                'entry_motion': out_motions,
                'entry_length': out_lengths,
                'entry_caption': out_captions_list,
                'entry_tokens': out_tokens_list,
            }

        print(f"Expanding eval entries with batched map (num_proc={NUM_PROC})...")
        expanded = hf_dataset.map(
            _expand_to_entries,
            batched=True,
            batch_size=256,
            remove_columns=hf_dataset.column_names,
            num_proc=NUM_PROC,
            desc=f"Expanding {split_name} data",
        )

        # 构建 data_dict
        data_dict = {}
        new_name_list = []
        length_list = []

        for i in range(len(expanded)):
            row = expanded[i]
            name = row['entry_name']
            motion = np.array(row['entry_motion'], dtype=np.float32)
            m_length = row['entry_length']

            # 还原 text_data 列表
            captions = row['entry_caption'].split('@@')
            tokens_strs = row['entry_tokens'].split('@@')
            text_data = []
            for cap, tok_str in zip(captions, tokens_strs):
                text_data.append({
                    'caption': cap,
                    'tokens': tok_str.split(' '),
                })

            if name not in data_dict:
                data_dict[name] = {
                    'motion': motion,
                    'length': m_length,
                    'text': text_data,
                }
                new_name_list.append(name)
                length_list.append(m_length)
            else:
                # 同名条目追加 text
                data_dict[name]['text'].extend(text_data)

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
        data_entry = self.data_dict[name]
        motion, m_length, text_list = data_entry['motion'], data_entry['length'], data_entry['text']

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
# 3. Tokenize 用 Dataset — 直接用 filter，无需手动循环
# ==========================================
class HF_TokenizeDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length=4, cache_dir=None):
        self.dataset_name = dataset_name
        self.unit_length = unit_length

        if dataset_name == 't2m':
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            self.min_motion_len = 40
        elif dataset_name == 'kit':
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            self.min_motion_len = 24

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        print(f"Loading {dataset_name} Train dataset from HuggingFace for tokenization...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)

        min_len = self.min_motion_len
        self.dataset = hf_dataset.filter(
            lambda meta: min_len <= meta['num_frames'] < 200,
            input_columns=['meta_data'],
            num_proc=NUM_PROC,
            desc="Filtering tokenize data",
        )
        self.dataset = self.dataset.with_format("numpy")
        print(f"HF Tokenize Dataset Loaded! Total valid motions: {len(self.dataset)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_item = self.dataset[item]
        motion = data_item['motion'].astype(np.float32)
        name = data_item['meta_data']['name']
        m_length = len(motion)

        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]
        motion = (motion - self.mean) / self.std
        return motion, name


def tokenize_collate_fn(batch):
    motions, names = zip(*batch)
    motion_tensor = torch.from_numpy(np.array(motions))
    return motion_tensor, names


# ==========================================
# 4. Transformer 训练用 Dataset — 高效版
#    关键优化：用 datasets.map() 并行展开，
#    避免 23k 次 for 循环 + 文件 I/O
# ==========================================
class HF_Text2MotionTokenDataset(data.Dataset):
    def __init__(self, dataset_name, codebook_size=1024, tokenizer_name=None,
                 unit_length=4, cache_dir=None, vq_dir=None):
        self.max_length = 64
        self.pointer = 0
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

        if vq_dir is not None:
            self.vq_dir = vq_dir
        else:
            data_root = "./dataset/KIT-ML" if dataset_name == 'kit' else "./dataset/HumanML3D"
            self.vq_dir = pjoin(data_root, tokenizer_name) if tokenizer_name else pjoin(data_root, 'vq_tokens')

        print(f"Loading {dataset_name} Train dataset from HuggingFace for Transformer training...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)
        print(f"HF raw train set size: {len(hf_dataset)}")

        # --- 预扫描可用的 vq token 文件名 (一次性 I/O) ---
        print("Scanning available VQ token files...")
        if os.path.isdir(self.vq_dir):
            available_names = set(
                f[:-4] for f in os.listdir(self.vq_dir) if f.endswith('.npy')
            )
        else:
            available_names = set()
        print(f"Found {len(available_names)} VQ token files in {self.vq_dir}")

        # --- 第一步：filter 掉没有 token 文件的样本 ---
        hf_dataset = hf_dataset.filter(
            lambda meta: meta['name'] in available_names,
            input_columns=['meta_data'],
            num_proc=NUM_PROC,
            desc="Filtering by available VQ tokens",
        )
        print(f"After filtering: {len(hf_dataset)} samples have VQ tokens")

        # --- 第二步：预加载所有 token 到内存 (向量化读取) ---
        print("Pre-loading all VQ tokens into memory...")
        token_cache = {}
        names_to_load = set()
        for i in range(len(hf_dataset)):
            names_to_load.add(hf_dataset[i]['meta_data']['name'])
        for name in names_to_load:
            token_path = pjoin(self.vq_dir, f'{name}.npy')
            try:
                token_cache[name] = np.load(token_path)
            except:
                pass
        print(f"Loaded {len(token_cache)} token arrays into memory")

        # --- 第三步：用 batched map 展开所有条目 ---
        _vq_dir = self.vq_dir
        _fps = fps
        _unit_length = unit_length

        def _expand_entries(batch):
            out_captions = []
            out_token_indices = []  # 存储序列化后的 token 信息
            out_token_names = []  # 用于后续从 cache 读取

            for idx in range(len(batch['caption'])):
                name = batch['meta_data'][idx]['name']
                raw_text_str = batch['caption'][idx]

                if name not in token_cache:
                    continue

                m_token_list = token_cache[name]
                parsed = _parse_text_lines(raw_text_str)
                if not parsed:
                    continue

                whole_captions = []
                has_whole = False

                for td in parsed:
                    if td['f_tag'] == 0.0 and td['to_tag'] == 0.0:
                        has_whole = True
                        whole_captions.append(td['caption'])
                    else:
                        f_i = int(td['f_tag'] * _fps / _unit_length)
                        t_i = int(td['to_tag'] * _fps / _unit_length)
                        if f_i >= t_i:
                            continue
                        sub_name = f"{name}_{td['f_tag']}_{td['to_tag']}"
                        # 存储切片信息
                        out_captions.append(td['caption'])
                        out_token_names.append(f"{name}|{f_i}|{t_i}")

                if has_whole:
                    for cap in whole_captions:
                        out_captions.append(cap)
                        out_token_names.append(f"{name}|WHOLE")

            return {
                'entry_caption': out_captions,
                'entry_token_ref': out_token_names,
            }

        print(f"Expanding Transformer train entries with batched map...")
        expanded = hf_dataset.map(
            _expand_entries,
            batched=True,
            batch_size=512,
            remove_columns=hf_dataset.column_names,
            num_proc=1,  # 因为访问 token_cache 闭包, 用 1 proc 避免序列化开销
            desc="Expanding train entries",
        )
        print(f"Expanded to {len(expanded)} entries")

        # --- 第四步：构建最终的列表 (极快，纯内存操作) ---
        self.entries = []
        for i in range(len(expanded)):
            row = expanded[i]
            caption = row['entry_caption']
            ref = row['entry_token_ref']

            parts = ref.split('|')
            name = parts[0]
            if name not in token_cache:
                continue

            m_token_list = token_cache[name]

            if parts[1] == 'WHOLE':
                # 整段
                if m_token_list.ndim == 1:
                    tokens = m_token_list
                else:
                    tokens = m_token_list[random.randint(0, len(m_token_list) - 1)]
            else:
                f_i, t_i = int(parts[1]), int(parts[2])
                if m_token_list.ndim == 1:
                    tokens = m_token_list[f_i:t_i]
                else:
                    tokens = m_token_list[0][f_i:t_i]  # 取第一个 quantizer

            if len(tokens) == 0:
                continue

            self.entries.append({
                'caption': caption,
                'm_tokens': tokens.copy(),
            })

        # 释放大缓存
        del token_cache
        del expanded

        print(f"HF Transformer Train Dataset Loaded! Total entries: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        entry = self.entries[item]
        m_tokens = entry['m_tokens'].copy()
        caption = entry['caption']

        # 随机 drop 一个 token
        coin = np.random.choice([False, False, True])
        if coin:
            if np.random.choice([True, False]):
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = len(m_tokens)

        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1,), dtype=int) * self.mot_end_idx,
                np.ones((self.max_motion_length - 1 - m_tokens_len,), dtype=int) * self.mot_pad_idx
            ], axis=0)
        else:
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1,), dtype=int) * self.mot_end_idx
            ], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len


# ==========================================
# 统一的 DataLoader 获取接口
# ==========================================

def get_train_loader(dataset_name, batch_size, window_size=64, unit_length=4,
                     num_workers=8, cache_dir=None):
    train_set = HF_VQMotionDataset(dataset_name, window_size=window_size,
                                   unit_length=unit_length, cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )


def get_val_loader(dataset_name, batch_size, w_vectorizer, is_test=False,
                   unit_length=4, num_workers=8, cache_dir=None):
    val_set = HF_Text2MotionDataset(dataset_name, is_test=is_test,
                                    w_vectorizer=w_vectorizer, unit_length=unit_length,
                                    cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True
    )


def get_tokenize_loader(dataset_name, batch_size=1, unit_length=4,
                        num_workers=0, cache_dir=None):
    token_set = HF_TokenizeDataset(dataset_name, unit_length=unit_length, cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        token_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=tokenize_collate_fn, drop_last=False
    )


def get_trans_train_loader(dataset_name, batch_size, codebook_size, tokenizer_name=None,
                           unit_length=4, num_workers=8, cache_dir=None, vq_dir=None):
    train_set = HF_Text2MotionTokenDataset(
        dataset_name, codebook_size=codebook_size,
        tokenizer_name=tokenizer_name, unit_length=unit_length,
        cache_dir=cache_dir, vq_dir=vq_dir
    )
    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )