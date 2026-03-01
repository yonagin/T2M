import os
import torch
from torch.utils import data
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# ==========================================
# 专门为 VQ-VAE 训练设计的 HF Dataset
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
        print(f"HF Dataset Loaded! Total valid motions: {len(self.dataset)}")

    def inv_transform(self, data_in):
        return data_in * self.std + self.mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_item = self.dataset[item]
        motion = data_item['motion'] 
        
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx : idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return motion


# ==========================================
# 专门为 VQ-VAE 评测设计的 HF Dataset（修复版）
# ==========================================
class HF_Text2MotionDataset(data.Dataset):
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

        # ============================================================
        # 核心：遍历 HF 数据集，还原原始的 data_dict 构建逻辑
        # 包含 f_tag/to_tag 子动作切片
        # ============================================================
        data_dict = {}
        new_name_list = []
        length_list = []

        for i in tqdm(range(len(hf_dataset)), desc="Building data_dict"):
            try:
                data_item = hf_dataset[i]
                motion = np.array(data_item['motion'], dtype=np.float32)
                name = data_item['meta_data']['name']

                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                # 解析 caption 字段（多行文本用 \n 连接）
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
                        # 子动作切片逻辑
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
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)

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

        # 按长度排序（与原始逻辑一致）
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

        # Randomly select a caption
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

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name

# ==========================================
# 统一的 DataLoader 获取接口
# ==========================================
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

def get_val_loader(dataset_name, batch_size, w_vectorizer, unit_length=4, num_workers=8, cache_dir=None):
    val_set = HF_Text2MotionDataset(dataset_name, is_test=False, w_vectorizer=w_vectorizer, unit_length=unit_length, cache_dir=cache_dir)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn, 
        drop_last=True
    )
    return val_loader