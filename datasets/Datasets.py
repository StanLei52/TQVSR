"""
Dataset for clip model
"""
import os
import logging
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import random
import h5py
from tqdm import tqdm
from easydict import EasyDict as edict
import sys
sys.path.append(".")
from AQVSR.utils.basic_utils import load_jsonl, load_json, load_from_feature_package

def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_image_text_alignment(sparse_alignment: list, max_img_feat: int, padding_value: int):
    """
    sparse_alignment: 
    max_img_feat: 
    return: N_img x max_img_feat x max_alignment_length 
            B     x max_img_feat x max_alignment_length

    sparse_alignment:
    [
        [   # image 1
            [1,2,3],    # whole image feature to the position of text feature embedding
            [4,5,6,7],  # bbox1 feature to the position of text feature embedding
            [8,9,10],   # bbox2 feature to the position of text feature embedding
        ],
         ...
         [  #image 2
            [1,2,3,4],
            [3,4,5,6,7],
            [8,9,10],
         ],
    ]
    ##
    Giving a sparse alignment matrix, return a dense one with padding; 
    """

    max_alignment_length = max([len(region_i)  for image_i in sparse_alignment for region_i in image_i])
    bs = len(sparse_alignment)
    padded_image_text_alignment = \
        np.ones((bs, max_img_feat, max_alignment_length), dtype=np.int32) * padding_value
    for i, img_ in enumerate(sparse_alignment):
        for j, feat_ in enumerate(img_):
            padded_image_text_alignment[i,j,:len(feat_)] = feat_
    
    return padded_image_text_alignment


def collate_concat_segment(batch,mask_ctx_text=False, mask_ctx_vis=False):
    batch_collect = edict()
    for key in batch[0].keys():
        batch_collect[key] = [item[key] for item in batch]
    pad_ctx_text_feat, ctx_text_mask = pad_sequences_1d(batch_collect.ctx_text_feat, dtype=torch.float)
    pad_ctx_vis_feat, ctx_vis_mask = pad_sequences_1d(batch_collect.ctx_vis_feat, dtype=torch.float)

    if mask_ctx_text: # mask transcript of video
        pad_ctx_text_feat = torch.zeros_like(pad_ctx_text_feat)
        ctx_text_mask = torch.zeros_like(ctx_text_mask)
    if mask_ctx_vis: # mask vision of video
        pad_ctx_vis_feat = torch.zeros_like(pad_ctx_vis_feat)
        ctx_vis_mask = torch.zeros_like(ctx_vis_mask)

    return edict(
        seg_id=batch_collect.seg_id,
        seg_name=batch_collect.seg_name,
        vid_name=batch_collect.vid_name,
        pad_ctx_vis_feat=pad_ctx_vis_feat,
        pad_ctx_text_feat=pad_ctx_text_feat,
        ctx_text_mask = ctx_text_mask,
        ctx_vis_mask = ctx_vis_mask
    )

def collate_for_concat_fusion(batch, 
    mask_query_text=False,
    mask_query_img=False,
    mask_ctx_text=False,
    mask_ctx_vis=False
):
    # collate function for concat text embedding and vis embedding
    batch_collect = edict()
    for key in batch[0].keys():
        batch_collect[key] = [item[key] for item in batch]

    pad_query_text_feat, query_text_mask = pad_sequences_1d(batch_collect.query_text_feat, dtype=torch.float)
    pad_query_vis_feat, query_vis_mask = pad_sequences_1d(batch_collect.query_vis_feat, dtype=torch.float)
    max_len_img_feat = pad_query_vis_feat.shape[1]
    pad_img_text_alignment = torch.from_numpy(
        pad_image_text_alignment(batch_collect.image_2_text_alignment, max_len_img_feat, padding_value=-1)
    )
    pad_ctx_text_feat, ctx_text_mask = pad_sequences_1d(batch_collect.ctx_text_feat, dtype=torch.float)
    pad_ctx_vis_feat, ctx_vis_mask = pad_sequences_1d(batch_collect.ctx_vis_feat, dtype=torch.float)

    if mask_query_text:
        pad_query_text_feat = torch.zeros_like(pad_query_text_feat)
        query_text_mask = torch.zeros_like(query_text_mask)
    if mask_query_img:
        pad_query_vis_feat = torch.zeros_like(pad_query_vis_feat)
        query_vis_mask = torch.zeros_like(query_vis_mask)
    if mask_ctx_text:
        pad_ctx_text_feat = torch.zeros_like(pad_ctx_text_feat)
        ctx_text_mask = torch.zeros_like(ctx_text_mask)
    if mask_ctx_vis:
        pad_ctx_vis_feat = torch.zeros_like(pad_ctx_vis_feat)
        ctx_vis_mask = torch.zeros_like(ctx_vis_mask)

    return edict(
        meta = batch_collect.meta,
        pad_query_text_feat = pad_query_text_feat,
        query_text_mask = query_text_mask,
        pad_query_vis_feat = pad_query_vis_feat,
        query_vis_mask = query_vis_mask,
        image_2_text_alignment = pad_img_text_alignment,
        pad_ctx_text_feat = pad_ctx_text_feat,
        pad_ctx_vis_feat = pad_ctx_vis_feat,
        ctx_text_mask = ctx_text_mask,
        ctx_vis_mask = ctx_vis_mask
    )

def collate_for_concat_MarginRanking(batch):
    # collate function for concat text embedding and vis embedding
    batch_collect = edict()
    for key in batch[0].keys():
        batch_collect[key] = [item[key] for item in batch]

    pad_query_text_feat, query_text_mask = pad_sequences_1d(batch_collect.query_text_feat, dtype=torch.float)
    pad_query_vis_feat, query_vis_mask = pad_sequences_1d(batch_collect.query_vis_feat, dtype=torch.float)
    max_len_img_feat = pad_query_vis_feat.shape[1]
    pad_img_text_alignment = torch.from_numpy(
        pad_image_text_alignment(batch_collect.image_2_text_alignment, max_len_img_feat, padding_value=-1)
    )
    pad_pos_ctx_text_feat, pos_ctx_text_mask = pad_sequences_1d(batch_collect.pos_ctx_text_feat, dtype=torch.float)
    pad_pos_ctx_vis_feat, pos_ctx_vis_mask = pad_sequences_1d(batch_collect.pos_ctx_vis_feat, dtype=torch.float)
    pad_intra_neg_ctx_text_feat, intra_neg_ctx_text_mask = pad_sequences_1d(batch_collect.intra_neg_ctx_text_feat, dtype=torch.float)
    pad_intra_neg_ctx_vis_feat, intra_neg_ctx_vis_mask = pad_sequences_1d(batch_collect.intra_neg_ctx_vis_feat, dtype=torch.float)
    pad_inter_neg_ctx_text_feat, inter_neg_ctx_text_mask = pad_sequences_1d(batch_collect.inter_neg_ctx_text_feat, dtype=torch.float)
    pad_inter_neg_ctx_vis_feat, inter_neg_ctx_vis_mask = pad_sequences_1d(batch_collect.inter_neg_ctx_vis_feat, dtype=torch.float)


    return edict(
        meta = batch_collect.meta,
        pad_query_text_feat = pad_query_text_feat,
        query_text_mask = query_text_mask,
        pad_query_vis_feat = pad_query_vis_feat,
        query_vis_mask = query_vis_mask,
        image_2_text_alignment = pad_img_text_alignment,
        
        pad_pos_ctx_text_feat = pad_pos_ctx_text_feat,
        pad_pos_ctx_vis_feat = pad_pos_ctx_vis_feat,
        pos_ctx_text_mask = pos_ctx_text_mask,
        pos_ctx_vis_mask = pos_ctx_vis_mask,
        
        pad_intra_neg_ctx_text_feat = pad_intra_neg_ctx_text_feat,
        pad_intra_neg_ctx_vis_feat = pad_intra_neg_ctx_vis_feat,
        intra_neg_ctx_text_mask = intra_neg_ctx_text_mask,
        intra_neg_ctx_vis_mask = intra_neg_ctx_vis_mask,
        
        pad_inter_neg_ctx_text_feat = pad_inter_neg_ctx_text_feat,
        pad_inter_neg_ctx_vis_feat = pad_inter_neg_ctx_vis_feat,
        inter_neg_ctx_text_mask = inter_neg_ctx_text_mask,
        inter_neg_ctx_vis_mask = inter_neg_ctx_vis_mask
    )


def collate_for_adding_fusion(batch):
    # collate function for adding text embedding and vis embedding for fusion
    batch_collect = edict()
    for key in batch[0].keys():
        batch_collect[key] = [item[key] for item in batch]
    pad_query_text_feat, query_text_mask = pad_sequences_1d(batch_collect.query_text_feat, dtype=torch.float)
    pad_query_vis_feat = torch.zeros(
        pad_query_text_feat.size()[:2] + (batch_collect.query_vis_feat[0].shape[-1],), 
        dtype=pad_query_text_feat.dtype
    )
    query_vis_mask = copy.deepcopy(query_text_mask)
    query_token_type_ids = torch.ones(
        pad_query_text_feat.shape[:2], dtype=torch.long
    )
    for bidx, (vis_feat, i2t) in enumerate(zip(batch_collect.query_vis_feat, batch_collect.image_2_text_alignment)):
        for idx,region2pos in enumerate(i2t):
            pad_query_vis_feat[bidx][region2pos] = vis_feat[idx]
            if idx==0: # 0 stands for the whole image
                query_token_type_ids[bidx][region2pos] = 0

    pad_ctx_text_feat, ctx_text_mask = pad_sequences_1d(batch_collect.ctx_text_feat, dtype=torch.float)
    pad_ctx_vis_feat, ctx_vis_mask = pad_sequences_1d(batch_collect.ctx_vis_feat, dtype=torch.float)

    return edict(
        meta = batch_collect.meta,
        pad_query_text_feat = pad_query_text_feat,
        query_text_mask = query_text_mask,
        pad_query_vis_feat = pad_query_vis_feat,
        query_vis_mask = query_vis_mask,
        query_token_type_ids = query_token_type_ids,
        image_2_text_alignment = batch_collect.image_2_text_alignment,
        pad_ctx_text_feat = pad_ctx_text_feat,
        pad_ctx_vis_feat = pad_ctx_vis_feat,
        ctx_text_mask = ctx_text_mask,
        ctx_vis_mask = ctx_vis_mask
    )

"""
Dummy dataset for debug:
"""
class DummyDataset(Dataset):
    """
    Args:
        dset_name, str, ["AQVSR"]
        
    Return:
        a dict: {
            "meta": {
                "query_id": int,
                "text_query": str,                                  # purely text query
                "original_query": str,
                "query_image_path": str,
                "vid_name": str,                                    # youtube_id (11)
                "answer_segment_name" list[str]                  # name of segments: ["xtuiYd45q1W_segment1",...]
                "answer_segment_id": list[segment_id],              # unique_segment_id
                "answer_segment_info": list[[st,ed], ... [st,ed]]   # start_time, end_time of coresponding segment

                "sample_seg_id_for_training": int,              # sample one segment for training
                #####
            }
     
            "query_text_feat": torch.tensor, (L, D_q)                       # query feature
            "query_vis_feat": torch.tensor,  (n_region, 2048)               # image feature&region feature
            "image_2_text_alignment": list[list]                              # image to token alignment
            "ctx_vis_feat": torch.tensor, (n_clip_in_segment, dim_video)     # video feature
            "ctx_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)      # sub feature

        }
    """
    def __init__(self, dset_name="dummy", data_path="", query_bert_path_or_handler="", sub_feat_path_or_handler="", vid_feat_path_or_handler="", normalize_vfeat=True, normalize_tfeat=True):
        self.dset_name = dset_name
        self.data = np.arange(1000)  # load data self.data = load_func("data_path") # query -> 

        self.query_bert_path_or_handler = query_bert_path_or_handler
        # self.query_image_feat_path_or_handler
        self.sub_feat_path_or_handler = sub_feat_path_or_handler
        self.vid_fear_path_or_handler = vid_feat_path_or_handler

        # Should be loaded from h5py file
        # self.query_feat_h5 = load_func(...path_to)
        # self.sub_feat_h5 = load_func(...path_to)
        # self.vid_feat_h5 = load_func(...path_to)

        ##### Dummy dataset, for debug purpose
        self.query_min_len, self.query_max_len = 10, 20
        self.n_clip_min, self.n_clip_max = 20, 80
        self.n_region_min, self.n_region_max = 1, 3


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ########
        # load from annotation
        ########
        # item = self.data[index]
        # meta = edict(
        #     query_id = item["query_id"],
        #     text_query = item["text_query"],
        #     vid_name = item["vid_name"],
        #     answer_segment_name = item["answer_segment_name"],
        #     answer_segment_id = item["answer_segment_info"],
        # )
        # 
        # query_text_feat = load_func(...),
        # query_vis_feat = load_func(...),
        # text_image_alignment = load_func(...),
        # ctx_vis_feat = load_func(...),
        # ctx_text_feat = load_func(...),
        # 
        # return edict(
        #     meta = meta, 
        #    ...
        # )

        ### For debug purpose
        qid = np.random.randint(1000)  # dummy: sample from one of the 1000 data
        text_query = "This is a sample from dummy dataset." # for debug
        vid_name = 'xcvFGRT_O3Q'
        answer_segment_name = ['xcvFGRT_O3Q_seg1','xcvFGRT_O3Q_seg3']
        answer_segment_id = [10555,10557]
        answer_segment_info = [[80,125], [220,320]]

        meta = edict(
            query_id = qid,
            text_query = text_query,
            vid_name = vid_name,
            answer_segment_name = answer_segment_name,
            answer_segment_id = answer_segment_id,
            answer_segment_info = answer_segment_info
        )

        query_len = np.random.randint(self.query_min_len, self.query_max_len+1)
        query_text_feat = torch.randn(query_len, 768)

        n_img_region = np.random.randint(self.n_region_min, self.n_region_max+1)
        query_vis_feat = torch.randn(n_img_region, 2048)
        img_2_text_alignment = np.split(
            np.arange(query_len),
            np.sort(np.random.choice(np.arange(1,query_len-1), n_img_region-1, replace=False))
        )

        n_clip = np.random.randint(self.n_clip_min, self.n_clip_max+1)
        ctx_vis_feat = torch.randn(n_clip, 2048)
        ctx_text_feat = torch.randn(n_clip, 768)

        return edict(
            meta = meta, 
            query_text_feat = query_text_feat,
            query_vis_feat = query_vis_feat,
            image_2_text_alignment = img_2_text_alignment,
            ctx_vis_feat = ctx_vis_feat,
            ctx_text_feat = ctx_text_feat
        )



ANNOTATION_PACKAGE_ROOT = '/home/stan/ai_assistant/data'
FEATURE_PACKAGE_ROOT = '/home/stan/ai_assistant/data'
ID_FILE_ROOT = '/home/stan/ai_assistant/data'

class AQVSR_query(Dataset):
    """
    Args:
        dset_name, str, "train" or "test"
        avg_pooling, boolean, default = False, True for avg_pooling, False for max_pooling
    Return:
        a dict: {
            "meta": {
                "query_id": int,
                "text_query": str,                                  # purely text query
                "original_query": str,
                "query_image_path": str,
                "vid_name": str,                                    # youtube_id (11)
                "answer_segment_name": list[str],                   # name of segments: ["xtuiYd45q1W_segment1",...]
                "answer_segment_id": list[segment_id],              # unique_segment_id
                "answer_segment_info": list[[st,ed], ... [st,ed]],  # start_time, end_time of coresponding segment
                "sample_seg_id_for_training": int,                  # sample one segment for training
                #####
            }
            "query_text_feat": torch.tensor, (L, D_q)                       # query feature
            "query_vis_feat": torch.tensor,  (n_region, 2048)               # image feature&region feature
            "image_2_text_alignment": list[list]                              # image to token alignment
            "ctx_vis_feat": torch.tensor, (n_clip_in_segment, dim_video)     # video feature
            "ctx_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)      # sub feature
        }
    """

    def __init__(self, dset_name="train", query_bert_path_or_handler="", sub_feat_path_or_handler="",
                 vid_feat_path_or_handler="", normalize_vfeat=True, normalize_tfeat=True,
                 avg_pooling=False, annotation_root=ANNOTATION_PACKAGE_ROOT, feature_root=FEATURE_PACKAGE_ROOT):
        assert dset_name in ['train', 'valid', 'test'], "dset_name should be in 'train' 'valid' and 'test'"
        self.dset_name = dset_name
        if dset_name == 'train':
            self.data = load_jsonl(os.path.join(annotation_root, 'trainset.jsonl'))
        elif dset_name == 'valid':
            self.data = load_jsonl(os.path.join(annotation_root, 'validset.jsonl'))
        elif dset_name == 'test':
            self.data = load_jsonl(os.path.join(annotation_root, 'testset.jsonl'))

        self.query_bert_path_or_handler = query_bert_path_or_handler
        self.sub_feat_path_or_handler = sub_feat_path_or_handler
        self.vid_fear_path_or_handler = vid_feat_path_or_handler
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        if avg_pooling:
            self.pooling = 'avg_pooling'
        else:
            self.pooling = 'max_pooling'

        # Should be loaded from h5py file
        with h5py.File(os.path.join(feature_root, 'feature.hdf5'), 'r') as f:
            self.query_text_feat = load_from_feature_package(f['query_text_feature'])
            self.query_img_feat = load_from_feature_package(f['query_grid_feature'])
            self.sub_text_feat = load_from_feature_package(f['subtitle_text_feature'])
            self.video_vis_feat = load_from_feature_package(f['frame_grid_feature'])

        # Generate query type list
        self.query_type = dict(
            text=[],
            video=[],
            text_video=[]
        )

        for item in self.data:
            q_type = item['query_type']
            if q_type == 'Text Only':
                self.query_type['text'].append(item['query_id'])
            elif q_type == 'Video Only':
                self.query_type['video'].append(item['query_id'])
            else:
                self.query_type['text_video'].append(item['query_id'])

        # generate list that does not overlap with train set
        if dset_name == 'valid' or dset_name == 'test':
            self.not_in_train = []
            for item in self.data:
                if item['not_in_train']:
                    self.not_in_train.append(item['query_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sample_seg_idx = random.sample(range(len(item['answer_segment_id'])), 1)[0]
        meta = edict(
            query_id=item['query_id'],
            query_name=item['query_name'],
            text_query=item['text_query'],
            original_query=item['original_query'],
            query_img_path=item['query_img_path'],
            vid_name=item['vid_name'],
            answer_segment_name=item['answer_segment_name'],
            answer_segment_id=item['answer_segment_id'],
            answer_segment_info=item['answer_segment_info'],
            sample_seg_id_for_training=item['answer_segment_id'][sample_seg_idx],
            sample_seg_name_for_training=item['answer_segment_name'][sample_seg_idx]
        )

        query_text_feat = self.query_text_feat[item['vid_name']][item['query_name']]['feature'][0]
        img_2_text_alignment = self.query_text_feat[item['vid_name']][item['query_name']]['img_alignment']

        query_vis_feat = self.query_img_feat[item['vid_name']][item['query_img_path'].split('/')[-1]]

        ctx_vis_feat = self.video_vis_feat[item['vid_name']][item['answer_segment_name'][sample_seg_idx]][self.pooling]
        ctx_text_feat = self.sub_text_feat[item['vid_name']][item['answer_segment_name'][sample_seg_idx]][self.pooling]

        if self.normalize_tfeat:
            query_text_feat = l2_normalize_np_array(query_text_feat)
            ctx_text_feat = l2_normalize_np_array(ctx_text_feat)

        if self.normalize_vfeat:
            query_vis_feat = l2_normalize_np_array(query_vis_feat)
            ctx_vis_feat = l2_normalize_np_array(ctx_vis_feat)

        return edict(
            meta=meta,
            query_text_feat=torch.from_numpy(query_text_feat),
            query_vis_feat=torch.from_numpy(query_vis_feat),
            image_2_text_alignment=img_2_text_alignment,
            ctx_vis_feat=torch.from_numpy(ctx_vis_feat),
            ctx_text_feat=torch.from_numpy(ctx_text_feat)
        )


class AQVSR_segment(Dataset):
    def __init__(self, dset_name="train", normalize_vfeat=True, normalize_tfeat=True,
                 avg_pooling=False, annotation_root=ANNOTATION_PACKAGE_ROOT, feature_root=FEATURE_PACKAGE_ROOT):
        assert dset_name in ['train', 'valid', 'test'], "dset_name should be in 'train' 'valid' and 'test'"
        self.dset_name = dset_name
        if dset_name == 'train':
            self.data = load_jsonl(os.path.join(annotation_root, 'trainset.jsonl'))
        elif dset_name == 'valid':
            self.data = load_jsonl(os.path.join(annotation_root, 'validset.jsonl'))
        elif dset_name == 'test':
            self.data = load_jsonl(os.path.join(annotation_root, 'testset.jsonl'))

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        if avg_pooling:
            self.pooling = 'avg_pooling'
        else:
            self.pooling = 'max_pooling'

        # Generate iterable segment list
        self.segment_list = []
        vid_set = set()
        for query in self.data:
            vid = query['query_name'][:11]
            vid_set.add(vid)

        seg2id = load_json(os.path.join(ID_FILE_ROOT, 'id.json'))['seg2id']
        for seg_name, seg_id in seg2id.items():
            vid = seg_name[:11]
            if vid in vid_set:
                self.segment_list.append([seg_id, seg_name, vid])

        # Should be loaded from h5py file
        with h5py.File(os.path.join(feature_root, 'feature.hdf5'), 'r') as f:
            self.sub_text_feat = load_from_feature_package(f['subtitle_text_feature'])
            self.video_vis_feat = load_from_feature_package(f['frame_grid_feature'])

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, index):
        seg = self.segment_list[index]
        seg_id = seg[0]
        seg_name = seg[1]
        vid = seg[2]

        ctx_vis_feat = self.video_vis_feat[vid][seg_name][self.pooling]
        ctx_text_feat = self.sub_text_feat[vid][seg_name][self.pooling]

        if self.normalize_tfeat:
            ctx_text_feat = l2_normalize_np_array(ctx_text_feat)

        if self.normalize_vfeat:
            ctx_vis_feat = l2_normalize_np_array(ctx_vis_feat)

        return edict(
            seg_id=seg_id,
            seg_name=seg_name,
            vid_name=vid,
            ctx_vis_feat=torch.from_numpy(ctx_vis_feat),
            ctx_text_feat=torch.from_numpy(ctx_text_feat)
        )


# Return format according to ranking loss
# pos, intra-neg, inter-neg
class AQVSR_Ranking(Dataset):
    """
    Args:
        avg_pooling, boolean, default = False, True for avg_pooling, False for max_pooling
    Return:
        a dict: {
            "meta": {
                "query_id": int,
                "text_query": str,                                  # purely text query
                "original_query": str,
                "query_image_path": str,
                "vid_name": str,                                    # youtube_id (11)
                "answer_segment_name": list[str],                   # name of segments: ["xtuiYd45q1W_segment1",...]
                "answer_segment_id": list[segment_id],              # unique_segment_id
                "answer_segment_info": list[[st,ed], ... [st,ed]],  # start_time, end_time of coresponding segment
                #   modified in v2:
                "pos_seg_id_for_training": int,                  # sample one ground truth segment for training
                "pos_seg_name_for_training": str,
                "intra_neg_seg_id_for_training": int,                  # sample one intra wrong segment for training
                "intra_neg_seg_name_for_training": str,
                "inter_neg_seg_id_for_training": int,                  # sample one inter wrong segment for training
                "inter_neg_seg_name_for_training": str,
            }
            "query_text_feat": torch.tensor, (L, D_q)                       # query feature
            "query_vis_feat": torch.tensor,  (n_region, 2048)               # image feature&region feature
            "image_2_text_alignment": list[list]                              # image to token alignment
            #   modified in v2:                                             # n_sample sub/video feature include the groundtruth
            "pos_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)
            "intra_neg_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)
            "inter_neg_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)
            "pos_vis_feat": torch.tensor, (n_sample, n_clip_in_segment, dim_video)
            "intra_neg_vis_feat": torch.tensor, (n_clip_in_segment, dim_video)
            "inter_neg_vis_feat": torch.tensor, (n_clip_in_segment, dim_video)
        }
    """

    def __init__(self, dset_name='train', normalize_vfeat=True, normalize_tfeat=True,
                 avg_pooling=False, annotation_root=ANNOTATION_PACKAGE_ROOT, feature_root=FEATURE_PACKAGE_ROOT):

        assert dset_name in ['train', 'valid', 'test'], "dset_name should be in 'train' 'valid' and 'test'"
        self.dset_name = dset_name
        if dset_name == 'train':
            self.data = load_jsonl(os.path.join(annotation_root, 'trainset.jsonl'))
        elif dset_name == 'valid':
            self.data = load_jsonl(os.path.join(annotation_root, 'validset.jsonl'))
        elif dset_name == 'test':
            self.data = load_jsonl(os.path.join(annotation_root, 'testset.jsonl'))

        # return dict should also be modified if change the neg number
        self.n_pos = 1
        self.n_neg_intra = 1
        self.n_neg_inter = 1

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        if avg_pooling:
            self.pooling = 'avg_pooling'
        else:
            self.pooling = 'max_pooling'

        # Generate iterable segment list, split segment to train/test set
        self.segment_list = []
        vid_set = set()
        for query in self.data:
            vid = query['query_name'][:11]
            vid_set.add(vid)

        seg2id = load_json(os.path.join(ID_FILE_ROOT, 'id.json'))['seg2id']
        for seg_name, seg_id in seg2id.items():
            vid = seg_name[:11]
            if vid in vid_set:
                self.segment_list.append([seg_id, seg_name, vid])

        # Should be loaded from h5py file
        with h5py.File(os.path.join(feature_root, 'feature.hdf5'), 'r') as f:
            self.query_text_feat = load_from_feature_package(f['query_text_feature'])
            self.query_img_feat = load_from_feature_package(f['query_grid_feature'])
            self.sub_text_feat = load_from_feature_package(f['subtitle_text_feature'])
            self.video_vis_feat = load_from_feature_package(f['frame_grid_feature'])

        # Add negative list
        for item_idx in range(len(self.data)):
            item = self.data[item_idx]

            negative_seg_id_intra = []
            negative_seg_id_inter = []
            negative_seg_name_intra = []
            negative_seg_name_inter = []
            for [seg_id, seg_name, vid] in self.segment_list:
                if seg_name in item['answer_segment_name']:
                    continue
                else:
                    if vid == item['vid_name']:
                        negative_seg_id_intra.append(seg_id)
                        negative_seg_name_intra.append(seg_name)
                    else:
                        negative_seg_id_inter.append(seg_id)
                        negative_seg_name_inter.append(seg_name)

            self.data[item_idx]['intra_negative_segment_name'] = negative_seg_name_intra
            self.data[item_idx]['intra_negative_segment_id'] = negative_seg_id_intra
            self.data[item_idx]['inter_negative_segment_name'] = negative_seg_name_inter
            self.data[item_idx]['inter_negative_segment_id'] = negative_seg_id_inter

        # Generate query type list
        self.query_type = dict(
            text=[],
            video=[],
            text_video=[]
        )

        for item in self.data:
            q_type = item['query_type']
            if q_type == 'Text Only':
                self.query_type['text'].append(item['query_id'])
            elif q_type == 'Video Only':
                self.query_type['video'].append(item['query_id'])
            else:
                self.query_type['text_video'].append(item['query_id'])

        # generate list that does not overlap with train set
        if dset_name == 'valid' or dset_name == 'test':
            self.not_in_train = []
            for item in self.data:
                if item['not_in_train']:
                    self.not_in_train.append(item['query_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # sample positive and negative segment
        positive_seg_id = item['answer_segment_id']
        positive_seg_name = item['answer_segment_name']
        negative_seg_name_intra = item['intra_negative_segment_name']
        negative_seg_name_inter = item['inter_negative_segment_name']
        negative_seg_id_intra = item['intra_negative_segment_id']
        negative_seg_id_inter = item['inter_negative_segment_id']

        positive_idx = random.sample(range(len(positive_seg_name)), self.n_pos)
        negative_idx_intra = random.sample(range(len(negative_seg_name_intra)), self.n_neg_intra)
        negative_idx_inter = random.sample(range(len(negative_seg_name_inter)), self.n_neg_inter)

        positive_seg_id_sampled = [positive_seg_id[idx] for idx in positive_idx]
        negative_seg_id_intra_sampled = [negative_seg_id_intra[idx] for idx in negative_idx_intra]
        negative_seg_id_inter_sampled = [negative_seg_id_inter[idx] for idx in negative_idx_inter]

        positive_seg_name_sampled = [positive_seg_name[idx] for idx in positive_idx]
        negative_seg_name_intra_sampled = [negative_seg_name_intra[idx] for idx in negative_idx_intra]
        negative_seg_name_inter_sampled = [negative_seg_name_inter[idx] for idx in negative_idx_inter]

        meta = edict(
            query_id=item['query_id'],
            query_name=item['query_name'],
            text_query=item['text_query'],
            original_query=item['original_query'],
            query_img_path=item['query_img_path'],
            vid_name=item['vid_name'],
            answer_segment_name=item['answer_segment_name'],
            answer_segment_id=item['answer_segment_id'],
            answer_segment_info=item['answer_segment_info'],
            pos_seg_id=positive_seg_id_sampled[0],  # note that this [0] need all n_pos/n_neg = 1
            pos_seg_name=positive_seg_name_sampled[0],
            intra_neg_seg_id=negative_seg_id_intra_sampled[0],
            intra_neg_seg_name=negative_seg_name_intra_sampled[0],
            inter_neg_seg_id=negative_seg_id_inter_sampled[0],
            inter_neg_seg_name=negative_seg_name_inter_sampled[0]
        )

        query_text_feat = self.query_text_feat[item['vid_name']][item['query_name']]['feature'][0]
        img_2_text_alignment = self.query_text_feat[item['vid_name']][item['query_name']]['img_alignment']

        query_vis_feat = self.query_img_feat[item['vid_name']][item['query_img_path'].split('/')[-1]]

        ctx_vis_feat = [self.video_vis_feat[seg_name[:11]][seg_name][self.pooling] for seg_name in
                        positive_seg_name_sampled + negative_seg_name_intra_sampled + negative_seg_name_inter_sampled]

        ctx_text_feat = [self.sub_text_feat[seg_name[:11]][seg_name][self.pooling] for seg_name in
                         positive_seg_name_sampled + negative_seg_name_intra_sampled + negative_seg_name_inter_sampled]

        if self.normalize_tfeat:
            query_text_feat = l2_normalize_np_array(query_text_feat)
            for i in range(len(ctx_text_feat)):
                ctx_text_feat[i] = torch.from_numpy(l2_normalize_np_array(ctx_text_feat[i]))

        if self.normalize_vfeat:
            query_vis_feat = l2_normalize_np_array(query_vis_feat)
            for i in range(len(ctx_vis_feat)):
                ctx_vis_feat[i] = torch.from_numpy(l2_normalize_np_array(ctx_vis_feat[i]))

        return edict(
            meta=meta,
            query_text_feat=torch.from_numpy(query_text_feat),
            query_vis_feat=torch.from_numpy(query_vis_feat),
            image_2_text_alignment=img_2_text_alignment,
            pos_ctx_vis_feat=ctx_vis_feat[0],
            intra_neg_ctx_vis_feat=ctx_vis_feat[1],
            inter_neg_ctx_vis_feat=ctx_vis_feat[2],
            pos_ctx_text_feat=ctx_text_feat[0],
            intra_neg_ctx_text_feat=ctx_text_feat[1],
            inter_neg_ctx_text_feat=ctx_text_feat[2],
        )


class AQVSR_Ranking_enum(Dataset):
    """
    Args:
        avg_pooling, boolean, default = False, True for avg_pooling, False for max_pooling
    Return:
        a dict: {
            "meta": {
                "query_id": int,
                "text_query": str,                                  # purely text query
                "original_query": str,
                "query_image_path": str,
                "vid_name": str,                                    # youtube_id (11)
                "answer_segment_name": list[str],                   # name of segments: ["xtuiYd45q1W_segment1",...]
                "answer_segment_id": list[segment_id],              # unique_segment_id
                "answer_segment_info": list[[st,ed], ... [st,ed]],  # start_time, end_time of coresponding segment
                #   modified in v2:
                "seg_id_for_ranking": int,                  #
                "seg_name_for_ranking": str,
            }
            "query_text_feat": torch.tensor, (L, D_q)                       # query feature
            "query_vis_feat": torch.tensor,  (n_region, 2048)               # image feature&region feature
            "image_2_text_alignment": list[list]                              # image to token alignment
            #   modified in v2:
            "ctx_text_feat": torch.tensor, (n_clip_in_segment, dim_sub)     # sampled sub/video feature
            "ctx_vis_feat": torch.tensor, (n_sample, n_clip_in_segment, dim_video)
        }
    """

    def __init__(self, dset_name='test', normalize_vfeat=True, normalize_tfeat=True,
                 avg_pooling=False, annotation_root=ANNOTATION_PACKAGE_ROOT, feature_root=FEATURE_PACKAGE_ROOT):

        assert dset_name in ['train', 'valid', 'test'], "dset_name should be in 'train' 'valid' and 'test'"
        self.dset_name = dset_name
        if dset_name == 'train':
            self.data = load_jsonl(os.path.join(annotation_root, 'trainset.jsonl'))
        elif dset_name == 'valid':
            self.data = load_jsonl(os.path.join(annotation_root, 'validset.jsonl'))
        elif dset_name == 'test':
            self.data = load_jsonl(os.path.join(annotation_root, 'testset.jsonl'))

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        if avg_pooling:
            self.pooling = 'avg_pooling'
        else:
            self.pooling = 'max_pooling'

        # Generate iterable segment list, split segment to train/test set
        self.pairlist = []
        vid_set = set()
        for query in self.data:
            vid = query['query_name'][:11]
            vid_set.add(vid)

        seg2id = load_json(os.path.join(ID_FILE_ROOT, 'id.json'))['seg2id']
        # collect query and seg
        self.query_ids = [self.data[i]['query_id'] for i in range(len(self.data))]
        self.seg_ids = [v for k, v in seg2id.items() if k[:11] in vid_set]
        self.n_query = len(self.query_ids)
        self.n_seg = len(self.seg_ids)
        # print(self.n_query, self.n_seg)

        for query in self.data:
            for seg_name, seg_id in seg2id.items():
                vid = seg_name[:11]
                if vid in vid_set:
                    self.pairlist.append(dict(
                        query_item=query,
                        seg_name=seg_name,
                        seg_id=seg_id,
                        vid=vid
                    ))

        # Should be loaded from h5py file
        with h5py.File(os.path.join(feature_root, 'feature.hdf5'), 'r') as f:
            self.query_text_feat = load_from_feature_package(f['query_text_feature'])
            self.query_img_feat = load_from_feature_package(f['query_grid_feature'])
            self.sub_text_feat = load_from_feature_package(f['subtitle_text_feature'])
            self.video_vis_feat = load_from_feature_package(f['frame_grid_feature'])

        # Generate query type list
        self.query_type = dict(
            text=[],
            video=[],
            text_video=[]
        )

        for item in self.data:
            q_type = item['query_type']
            if q_type == 'Text Only':
                self.query_type['text'].append(item['query_id'])
            elif q_type == 'Video Only':
                self.query_type['video'].append(item['query_id'])
            else:
                self.query_type['text_video'].append(item['query_id'])

        # generate list that does not overlap with train set
        if dset_name == 'valid' or dset_name == 'test':
            self.not_in_train = []
            for item in self.data:
                if item['not_in_train']:
                    self.not_in_train.append(item['query_id'])

    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, index):
        pair = self.pairlist[index]
        item = pair['query_item']
        seg_name = pair['seg_name']
        seg_id = pair['seg_id']
        vid = pair['vid']
        meta = edict(
            query_id=item['query_id'],
            query_name=item['query_name'],
            text_query=item['text_query'],
            original_query=item['original_query'],
            query_img_path=item['query_img_path'],
            vid_name=item['vid_name'],
            answer_segment_name=item['answer_segment_name'],
            answer_segment_id=item['answer_segment_id'],
            answer_segment_info=item['answer_segment_info'],
            seg_id_for_ranking=seg_id,
            seg_name_for_ranking=seg_name
        )

        query_text_feat = self.query_text_feat[item['vid_name']][item['query_name']]['feature'][0]
        img_2_text_alignment = self.query_text_feat[item['vid_name']][item['query_name']]['img_alignment']

        query_vis_feat = self.query_img_feat[item['vid_name']][item['query_img_path'].split('/')[-1]]

        ctx_vis_feat = self.video_vis_feat[vid][seg_name][self.pooling]
        ctx_text_feat = self.sub_text_feat[vid][seg_name][self.pooling]

        if self.normalize_tfeat:
            query_text_feat = l2_normalize_np_array(query_text_feat)
            ctx_text_feat = l2_normalize_np_array(ctx_text_feat)

        if self.normalize_vfeat:
            query_vis_feat = l2_normalize_np_array(query_vis_feat)
            ctx_vis_feat = l2_normalize_np_array(ctx_vis_feat)

        return edict(
            meta=meta,
            query_text_feat=torch.from_numpy(query_text_feat),
            query_vis_feat=torch.from_numpy(query_vis_feat),
            image_2_text_alignment=img_2_text_alignment,
            ctx_vis_feat=torch.from_numpy(ctx_vis_feat),
            ctx_text_feat=torch.from_numpy(ctx_text_feat)
        )

if __name__ == "__main__":
    pass