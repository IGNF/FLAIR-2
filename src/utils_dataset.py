import numpy as np
import yaml

import torch
import torch
from torch.nn import functional as F


def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def filter_dates(mask, clouds:bool=2, area_threshold:float=0.5, proba_threshold:int=60):
    """ Mask : array T*2*H*W
        Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
        Area_threshold : threshold on the surface covered by the clouds / snow 
        Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)
        Return array of indexes to keep
    """
    dates_to_keep = []
    
    for t in range(mask.shape[0]):
        if clouds != 2:
            cover = np.count_nonzero(mask[t, clouds, :,:]>=proba_threshold)
        else:
            cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
        cover /= mask.shape[2]*mask.shape[3]
        if cover < area_threshold:
            dates_to_keep.append(t)

    return dates_to_keep


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)



def pad_collate_train(dict, pad_value=0):
       
    _imgs   = [i['patch'] for i in dict]    
    _sen    = [i['spatch'] for i in dict] 
    _dates  = [i['dates'] for i in dict]
    _msks   = [i['labels'] for i in dict] 
    _smsks  = [i['slabels'] for i in dict]
    _mtd    = [i['mtd'] for i in dict]

    sizes = [e.shape[0] for e in _sen]
    m = max(sizes)
    padded_data, padded_dates = [],[]
    if not all(s == m for s in sizes):
        for data, date in zip(_sen, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data = _sen
        padded_dates = _dates
          
    batch = {
             "patch": torch.stack(_imgs, dim=0),
             "spatch": torch.stack(padded_data, dim=0),
             "dates": torch.stack(padded_dates, dim=0),
             "labels": torch.stack(_msks, dim=0),
             "slabels": torch.stack(_smsks, dim=0),
             "mtd" : torch.stack(_mtd, dim=0),
            }  
    return batch



def pad_collate_predict(dict, pad_value=0):
    
    _imgs   = [i['patch'] for i in dict]
    _sen    = [i['spatch'] for i in dict] 
    _dates  = [i['dates'] for i in dict]
    _mtd    = [i['mtd'] for i in dict]
    _ids   = [i['id'] for i in dict] 


    sizes = [e.shape[0] for e in _sen]
    m = max(sizes)
    padded_data, padded_dates = [],[]
    if not all(s == m for s in sizes):
        for data, date in zip(_sen, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data = _sen
        padded_dates = _dates
          
    batch = {
             "patch": torch.stack(_imgs, dim=0),
             "spatch": torch.stack(padded_data, dim=0),
             "dates": torch.stack(padded_dates, dim=0),
             "mtd" : torch.stack(_mtd, dim=0),
             "id": _ids,
            }  
    return batch