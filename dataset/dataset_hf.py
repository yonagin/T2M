# data_hf.py (真正高效版)

import os
import torch
from torch.utils import data
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate
from os.path import join as pjoin
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_PROC = min(8, os.cpu_count() or 1)


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def _parse_text_lines(raw_text_str):
    """解析 caption 字符串为结构化列表"""
    results = []
    for line in raw_text_str.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split('#')
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        results.append({
            'caption': parts[0],
            'tokens': parts[1].split(' '),
            'f_tag': f_tag,
            'to_tag': to_tag,
        })
    return results


def _load_one_npy(args):
    """线程池用：加载单个 npy 文件"""
    name, path = args
    try:
        return name, np.load(path)
    except:
        return name, None


def _bulk_load_npy(vq_dir, names):
    """
    多线程并行加载所有 npy 文件
    np.load 主要瓶颈是磁盘 I/O 等待，用线程池可以大幅加速
    """
    tasks = [(name, pjoin(vq_dir, f'{name}.npy')) for name in names]
    cache = {}
    # 线程池，不是进程池！I/O bound 用线程就够了
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_load_one_npy, t): t[0] for t in tasks}
        for future in futures:
            name, arr = future.result()
            if arr is not None:
                cache[name] = arr
    return cache


# ==========================================
# 1. VQ-VAE 训练用 Dataset
# ==========================================
class HF_VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4, cache_dir=None):
        self.window_size = window_size
        self.unit_length = unit_length

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
        print(f"HF VQ Train Dataset Loaded! Total: {len(self.dataset)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_item = self.dataset[item]
        motion = data_item['motion']
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx: idx + self.window_size]
        return (motion - self.mean) / self.std


# ==========================================
# 2. VQ-VAE 评测用 Dataset
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
        print(f"Loading {dataset_name} {split_name} from HuggingFace...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split=split_name, cache_dir=cache_dir)

        # eval 集通常很小(几百到几千)，直接循环没问题
        data_dict = {}
        new_name_list = []
        length_list = []

        for i in range(len(hf_dataset)):
            try:
                data_item = hf_dataset[i]
                motion = np.array(data_item['motion'], dtype=np.float32)
                name = data_item['meta_data']['name']

                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                parsed = _parse_text_lines(data_item['caption'])
                if not parsed:
                    continue

                whole_text = []
                has_whole = False

                for td in parsed:
                    if td['f_tag'] == 0.0 and td['to_tag'] == 0.0:
                        has_whole = True
                        whole_text.append(td)
                    else:
                        n_motion = motion[int(td['f_tag'] * fps): int(td['to_tag'] * fps)]
                        if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                            continue
                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in data_dict:
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        data_dict[new_name] = {
                            'motion': n_motion, 'length': len(n_motion),
                            'text': [{'caption': td['caption'], 'tokens': td['tokens']}]
                        }
                        new_name_list.append(new_name)
                        length_list.append(len(n_motion))

                if has_whole:
                    data_dict[name] = {
                        'motion': motion, 'length': len(motion),
                        'text': [{'caption': t['caption'], 'tokens': t['tokens']} for t in whole_text]
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print(f"HF Eval Dataset Loaded! entries: {len(self.data_dict)}, after pointer: {len(self)}")

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
        d = self.data_dict[name]
        motion, m_length, text_list = d['motion'], d['length'], d['text']

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
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name


# ==========================================
# 3. Tokenize 用 Dataset — 直接用 filter
# ==========================================
class HF_TokenizeDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length=4, cache_dir=None):
        self.unit_length = unit_length

        if dataset_name == 't2m':
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            min_len = 40
        elif dataset_name == 'kit':
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            min_len = 24

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        print(f"Loading {dataset_name} Train for tokenization...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)
        self.dataset = hf_dataset.filter(
            lambda meta: min_len <= meta['num_frames'] < 200,
            input_columns=['meta_data'],
            num_proc=NUM_PROC,
        )
        self.dataset = self.dataset.with_format("numpy")
        print(f"HF Tokenize Dataset Loaded! Total: {len(self.dataset)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_item = self.dataset[item]
        motion = data_item['motion'].astype(np.float32)
        name = data_item['meta_data']['name']
        m_length = (len(motion) // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]
        return (motion - self.mean) / self.std, name


def tokenize_collate_fn(batch):
    motions, names = zip(*batch)
    return torch.from_numpy(np.array(motions)), names


# ==========================================
# 4. Transformer 训练用 Dataset — 真正高效版
#    关键：多线程并行加载 npy + 纯内存构建
# ==========================================
class HF_Text2MotionTokenDataset(data.Dataset):
    def __init__(self, dataset_name, codebook_size=1024, tokenizer_name=None,
                 unit_length=4, cache_dir=None, vq_dir=None):
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

        print(f"Loading {dataset_name} Train from HuggingFace for Transformer...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)
        print(f"Raw train set: {len(hf_dataset)}")

        # Step 1: 获取所有可用的 token 文件名 (瞬间完成)
        print(f"Scanning {self.vq_dir} ...")
        available = set()
        if os.path.isdir(self.vq_dir):
            available = {f[:-4] for f in os.listdir(self.vq_dir) if f.endswith('.npy')}
        print(f"Found {len(available)} VQ token files")

        # Step 2: 先收集所有需要的 name (纯内存，秒级)
        needed_names = set()
        hf_names = []
        for i in range(len(hf_dataset)):
            name = hf_dataset[i]['meta_data']['name']
            hf_names.append(name)
            if name in available:
                needed_names.add(name)
        print(f"Need to load {len(needed_names)} token files")

        # Step 3: 多线程并行加载所有 npy (关键加速！)
        print(f"Loading VQ tokens with 32 threads...")
        token_cache = _bulk_load_npy(self.vq_dir, needed_names)
        print(f"Loaded {len(token_cache)} token arrays")

        # Step 4: 构建 entries (纯内存操作，极快)
        self.entries = []
        for i in range(len(hf_dataset)):
            name = hf_names[i]
            if name not in token_cache:
                continue

            m_token_list = token_cache[name]
            raw_text = hf_dataset[i]['caption']
            parsed = _parse_text_lines(raw_text)
            if not parsed:
                continue

            whole_captions = []
            has_whole = False

            for td in parsed:
                if td['f_tag'] == 0.0 and td['to_tag'] == 0.0:
                    has_whole = True
                    whole_captions.append(td['caption'])
                else:
                    f_i = int(td['f_tag'] * fps / unit_length)
                    t_i = int(td['to_tag'] * fps / unit_length)
                    if f_i >= t_i:
                        continue
                    if m_token_list.ndim == 1:
                        sub_tokens = m_token_list[f_i:t_i]
                    else:
                        sub_tokens = m_token_list[0][f_i:t_i]
                    if len(sub_tokens) == 0:
                        continue
                    self.entries.append({
                        'caption': td['caption'],
                        'm_tokens': sub_tokens.copy(),
                    })

            if has_whole:
                if m_token_list.ndim == 1:
                    whole_tokens = m_token_list
                else:
                    whole_tokens = m_token_list[0]
                for cap in whole_captions:
                    self.entries.append({
                        'caption': cap,
                        'm_tokens': whole_tokens.copy(),
                    })

        del token_cache
        print(f"Transformer Train Dataset ready! Total entries: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        entry = self.entries[item]
        m_tokens = entry['m_tokens'].copy()
        caption = entry['caption']

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
# DataLoader 接口
# ==========================================

def get_train_loader(dataset_name, batch_size, window_size=64, unit_length=4,
                     num_workers=8, cache_dir=None):
    train_set = HF_VQMotionDataset(dataset_name, window_size=window_size,
                                   unit_length=unit_length, cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)


def get_val_loader(dataset_name, batch_size, w_vectorizer, is_test=False,
                   unit_length=4, num_workers=8, cache_dir=None):
    val_set = HF_Text2MotionDataset(dataset_name, is_test=is_test,
                                    w_vectorizer=w_vectorizer, unit_length=unit_length,
                                    cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True)


def get_tokenize_loader(dataset_name, batch_size=1, unit_length=4,
                        num_workers=0, cache_dir=None):
    token_set = HF_TokenizeDataset(dataset_name, unit_length=unit_length, cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        token_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=tokenize_collate_fn, drop_last=False)


def get_trans_train_loader(dataset_name, batch_size, codebook_size, tokenizer_name=None,
                           unit_length=4, num_workers=8, cache_dir=None, vq_dir=None):
    train_set = HF_Text2MotionTokenDataset(
        dataset_name, codebook_size=codebook_size,
        tokenizer_name=tokenizer_name, unit_length=unit_length,
        cache_dir=cache_dir, vq_dir=vq_dir)
    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)