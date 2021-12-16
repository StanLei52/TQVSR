
import os
import json
import zipfile
import numpy as np
import pickle

import torch
from collections import OrderedDict
from itertools import repeat

class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def load_json(filename):
    with open(filename, "r") as f:
        return json.loads(f.readlines()[0].strip("\n"))

def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def load_from_feature_package(group_handle):
    feature_dict = dict()
    vids = group_handle.keys()
    for vid in vids:
        feature_dict[vid] = dict()
        sub_groups = group_handle[vid].keys()
        for sub_group in sub_groups:
            if '.jpg' in sub_group:
                regions = group_handle[vid][sub_group].keys()
                region_feature_list = [[] for r in regions]
                for region in regions:
                    if region == 'image':
                        region_feature_list[0] = group_handle[vid][sub_group][region][0].squeeze()
                    elif region == 'bbox' or region == 'box':
                        region_feature_list[1] = group_handle[vid][sub_group][region][0].squeeze()
                    else:
                        bbox_idx = int(region[4:])
                        region_feature_list[bbox_idx] = group_handle[vid][sub_group][region][0].squeeze()
                feature_dict[vid][sub_group] = np.array(region_feature_list)
            else:
                feature_dict[vid][sub_group] = dict()
                datas = group_handle[vid][sub_group].keys()
                for data in datas:
                    if data == 'img_alignment':
                        img_alignment_rows = group_handle[vid][sub_group][data].keys()
                        feature_dict[vid][sub_group][data] = [[] for i in img_alignment_rows]
                        for img_alignment_row in img_alignment_rows:
                            int(img_alignment_row)
                            feature_dict[vid][sub_group][data][int(img_alignment_row)] = \
                                group_handle[vid][sub_group][data][img_alignment_row][:].tolist()
                    elif data == 'token':
                        token_list = group_handle[vid][sub_group][data][:].tolist()
                        feature_dict[vid][sub_group][data] = [str(token)[2:-1] for token in token_list]
                    else:
                        if len(group_handle[vid][sub_group][data][:]) == 4:
                            feature_dict[vid][sub_group][data] = group_handle[vid][sub_group][data][:].squeeze()
                        else:
                            feature_dict[vid][sub_group][data] = group_handle[vid][sub_group][data][:]

    return feature_dict