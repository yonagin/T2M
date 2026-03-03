# data_hf.py (optimized)

import os
import torch
from torch.utils import data
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate
from os.path import join as pjoin
import hashlib


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# ==========================================
# 辅助：解析 caption 字符串
# ==========================================
def _parse_text_lines(raw_text_str):
    """解析 '#' 分隔的 caption 格式，返回 text_data 列表"""
    text_lines = [line for line in raw_text_str.split('\n') if line.strip() != '']
    text_data = []
    for line in text_lines:
        line_split = line.strip().split('#')
        if len(line_split) < 4:
            continue
        caption = line_split[0]
        tokens = line_split[1].split(' ')
        f_tag = float(line_split[2])
        to_tag = float(line_split[3])
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        text_data.append({
            'caption': caption,
            'tokens': tokens,
            'f_tag': f_tag,
            'to_tag': to_tag,
        })
    return text_data


def _get_meta_dir(dataset_name):
    if dataset_name == 't2m':
        return 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    elif dataset_name == 'kit':
        return 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _num_proc():
    return min(8, os.cpu_count() or 1)


# ==========================================
# 1. VQ-VAE 训练用 Dataset (motion only)
#    优化：用 .filter() 替代手动循环
# ==========================================
class HF_VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4, cache_dir=None):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.meta_dir = _get_meta_dir(dataset_name)

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        print(f"Loading {dataset_name} Train dataset from HuggingFace...")
        self.dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)

        # ✅ 用 datasets 原生 filter，多进程并行，无 Python 循环
        self.dataset = self.dataset.filter(
            lambda meta: meta['num_frames'] >= self.window_size,
            input_columns=['meta_data'],
            num_proc=_num_proc(),
        )

        # ✅ 设置输出格式为 numpy，__getitem__ 直接拿到 ndarray
        self.dataset = self.dataset.with_format("numpy")
        print(f"HF VQ Train Dataset Loaded! Total valid motions: {len(self.dataset)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        motion = self.dataset[item]['motion']
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx: idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return motion


# ==========================================
# 2. VQ-VAE 评测用 Dataset (motion + text)
#    优化：用 .map() 批量预处理 + .filter() 替代 for 循环
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
            self.meta_dir = _get_meta_dir(dataset_name)
            self.min_motion_len = 40
            self.fps = 20
        elif dataset_name == 'kit':
            self.max_motion_length = 196
            self.meta_dir = _get_meta_dir(dataset_name)
            self.min_motion_len = 24
            self.fps = 12.5

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        split_name = "test" if is_test else "val"
        print(f"Loading {dataset_name} {split_name} dataset from HuggingFace...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split=split_name, cache_dir=cache_dir)

        # ✅ 核心优化：用 .map() 批量展开，替代 Python for 循环
        min_motion_len = self.min_motion_len
        fps = self.fps

        def _expand_entries(batch):
            """
            批量处理函数：将每个样本展开为可能的多条记录。
            返回 flat lists 以供 datasets 构建新 dataset。
            """
            out_names = []
            out_motions = []
            out_lengths = []
            out_text_jsons = []  # 序列化的 text_data 列表

            for i in range(len(batch['caption'])):
                motion = np.array(batch['motion'][i], dtype=np.float32)
                name = batch['meta_data'][i]['name']
                raw_text_str = batch['caption'][i]

                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                text_entries = _parse_text_lines(raw_text_str)
                if not text_entries:
                    continue

                full_text_data = []
                for td in text_entries:
                    f_tag, to_tag = td['f_tag'], td['to_tag']
                    if f_tag == 0.0 and to_tag == 0.0:
                        full_text_data.append({
                            'caption': td['caption'],
                            'tokens': td['tokens']
                        })
                    else:
                        n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                        if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                            continue
                        # 子动作单独作为一条记录
                        sub_name = hashlib.md5(
                            f"{name}_{f_tag}_{to_tag}".encode()
                        ).hexdigest()[:8] + '_' + name
                        out_names.append(sub_name)
                        out_motions.append(n_motion.tolist())
                        out_lengths.append(len(n_motion))
                        out_text_jsons.append([{
                            'caption': td['caption'],
                            'tokens': ' '.join(td['tokens'])
                        }])

                if full_text_data:
                    out_names.append(name)
                    out_motions.append(motion.tolist())
                    out_lengths.append(len(motion))
                    out_text_jsons.append([{
                        'caption': t['caption'],
                        'tokens': ' '.join(t['tokens'])
                    } for t in full_text_data])

            return {
                'entry_name': out_names,
                'entry_motion': out_motions,
                'entry_length': out_lengths,
                'entry_texts': out_text_jsons,
            }

        # ✅ 批量 map，利用多进程展开
        expanded = hf_dataset.map(
            _expand_entries,
            batched=True,
            batch_size=256,
            remove_columns=hf_dataset.column_names,
            num_proc=_num_proc(),
            desc=f"Expanding eval data ({split_name})",
        )

        # ✅ 按 length 排序
        expanded = expanded.sort('entry_length')

        # 加载到内存构建 data_dict（评测集很小，直接取全部）
        self.data_dict = {}
        self.name_list = []
        self.length_list = []

        for i in range(len(expanded)):
            row = expanded[i]
            name = row['entry_name']
            motion = np.array(row['entry_motion'], dtype=np.float32)
            length = row['entry_length']
            text_data = [{
                'caption': t['caption'],
                'tokens': t['tokens'].split(' ')
            } for t in row['entry_texts']]

            self.data_dict[name] = {
                'motion': motion,
                'length': length,
                'text': text_data,
            }
            self.name_list.append(name)
            self.length_list.append(length)

        self.length_arr = np.array(self.length_list)
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
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name


# ==========================================
# 3. Tokenize 用 Dataset (编码 motion → token)
#    优化：用 .filter() + .map() 替代 for 循环
# ==========================================
class HF_TokenizeDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length=4, cache_dir=None):
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        self.meta_dir = _get_meta_dir(dataset_name)

        if dataset_name == 't2m':
            min_motion_len = 40
        elif dataset_name == 'kit':
            min_motion_len = 24

        self.mean = np.load(os.path.join(self.meta_dir, 'mean.npy'))
        self.std = np.load(os.path.join(self.meta_dir, 'std.npy'))

        print(f"Loading {dataset_name} Train dataset from HuggingFace for tokenization...")
        hf_dataset = load_dataset("TeoGchx/HumanML3D", split="train", cache_dir=cache_dir)

        # ✅ 向量化 filter：过滤过短/过长
        self.dataset = hf_dataset.filter(
            lambda meta: min_motion_len <= meta['num_frames'] < 200,
            input_columns=['meta_data'],
            num_proc=_num_proc(),
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
# 4. Transformer 训练用 Dataset (text + motion tokens)
#    优化：用 .map() 批量展开替代 for 循环
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

        vq_dir_local = self.vq_dir

        # ✅ 批量 map 展开，替代逐条 for 循环
        def _expand_token_entries(batch):
            out_names = []
            out_captions = []
            out_token_paths = []

            for i in range(len(batch['caption'])):
                name = batch['meta_data'][i]['name']
                token_path = pjoin(vq_dir_local, f'{name}.npy')
                if not os.path.exists(token_path):
                    continue

                raw_text_str = batch['caption'][i]
                text_entries = _parse_text_lines(raw_text_str)
                if not text_entries:
                    continue

                has_full = False
                full_captions = []

                for td in text_entries:
                    f_tag, to_tag = td['f_tag'], td['to_tag']
                    if f_tag == 0.0 and to_tag == 0.0:
                        has_full = True
                        full_captions.append(td['caption'])
                    else:
                        # 子动作
                        sub_name = f"{name}_{f_tag}_{to_tag}"
                        out_names.append(sub_name)
                        out_captions.append(td['caption'])
                        out_token_paths.append(token_path)

                if has_full:
                    # 全动作：存所有 full caption（用 ||| 分隔）
                    out_names.append(name)
                    out_captions.append('|||'.join(full_captions))
                    out_token_paths.append(token_path)

            return {
                'entry_name': out_names,
                'entry_caption': out_captions,
                'entry_token_path': out_token_paths,
            }

        expanded = hf_dataset.map(
            _expand_token_entries,
            batched=True,
            batch_size=512,
            remove_columns=hf_dataset.column_names,
            num_proc=_num_proc(),
            desc="Expanding Transformer train data",
        )

        # ✅ 现在从 expanded dataset 构建 data_dict（只读 token 文件一次）
        # 用 dict 缓存已加载的 token 文件避免重复 IO
        _token_cache = {}

        self.data_dict = {}
        self.name_list = []

        for i in range(len(expanded)):
            row = expanded[i]
            entry_name = row['entry_name']
            caption_str = row['entry_caption']
            token_path = row['entry_token_path']

            # 加载 token（缓存）
            if token_path not in _token_cache:
                _token_cache[token_path] = np.load(token_path)
            m_token_list = _token_cache[token_path]

            # 处理子动作 token 切片
            if '_' in entry_name and entry_name.split('_')[0] not in ('', entry_name):
                # 尝试解析 f_tag, to_tag
                parts = entry_name.rsplit('_', 2)
                if len(parts) == 3:
                    try:
                        f_tag = float(parts[1])
                        to_tag = float(parts[2])
                        m_token_list_new = [
                            tokens[int(f_tag * fps / unit_length): int(to_tag * fps / unit_length)]
                            for tokens in (m_token_list if m_token_list.ndim > 1 else [m_token_list])
                            if int(f_tag * fps / unit_length) < int(to_tag * fps / unit_length)
                        ]
                        if len(m_token_list_new) == 0:
                            continue
                        m_token_list = m_token_list_new
                    except (ValueError, IndexError):
                        pass

            # 解析 caption
            if '|||' in caption_str:
                captions = caption_str.split('|||')
            else:
                captions = [caption_str]

            if entry_name in self.data_dict:
                # 追加 caption
                self.data_dict[entry_name]['text'].extend(
                    [{'caption': c} for c in captions]
                )
            else:
                self.data_dict[entry_name] = {
                    'm_token_list': m_token_list if isinstance(m_token_list, list) else m_token_list,
                    'text': [{'caption': c} for c in captions],
                }
                self.name_list.append(entry_name)

        del _token_cache
        print(f"HF Transformer Train Dataset Loaded! Total valid entries: {len(self.data_dict)}")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data_entry = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data_entry['m_token_list'], data_entry['text']

        if isinstance(m_token_list, np.ndarray):
            m_tokens = random.choice(m_token_list) if m_token_list.ndim > 1 else m_token_list
        else:
            m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption = text_data['caption']

        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

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
# 统一的 DataLoader 获取接口（不变）
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
                                     w_vectorizer=w_vectorizer,
                                     unit_length=unit_length, cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True
    )


def get_tokenize_loader(dataset_name, batch_size=1, unit_length=4,
                         num_workers=0, cache_dir=None):
    token_set = HF_TokenizeDataset(dataset_name, unit_length=unit_length,
                                    cache_dir=cache_dir)
    return torch.utils.data.DataLoader(
        token_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=tokenize_collate_fn, drop_last=False
    )


def get_trans_train_loader(dataset_name, batch_size, codebook_size,
                            tokenizer_name=None, unit_length=4,
                            num_workers=8, cache_dir=None, vq_dir=None):
    train_set = HF_Text2MotionTokenDataset(
        dataset_name, codebook_size=codebook_size,
        tokenizer_name=tokenizer_name, unit_length=unit_length,
        cache_dir=cache_dir, vq_dir=vq_dir
    )
    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )