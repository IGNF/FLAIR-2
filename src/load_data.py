import os
import numpy as np
from pathlib import Path 
import json
from random import shuffle




def load_data (config: dict, val_percent=0.8): 
    """ Returns dicts (train/val/test) with 6 keys: 
    - PATH_IMG : aerial image (path, str) 
    - PATH_SP_DATA : satellite image (path, str) 
    - PATH_SP_DATES : satellite product names (path, str) 
    - PATH_SP_MASKS : satellite clouds / snow masks (path, str)
    - SP_COORDS : centroid coordinate of patch in superpatch (list, e.g., [56,85]) 
    - PATH_LABELS : labels (path, str) 
    - MTD_AERIAL: aerial images encoded metadata
    """ 
    def get_data_paths(config: dict, path_domains: str, paths_data: dict, matching_dict: dict, test_set: bool) -> dict: 
        #### return data paths 
        def list_items(path, filter): 
            for path in Path(path).rglob(filter): 
                yield path.resolve().as_posix() 
        status = ['train' if test_set == False else 'test'][0] 
        ## data paths dict
        data = {'PATH_IMG':[], 'PATH_SP_DATA':[], 'SP_COORDS':[], 'PATH_SP_DATES':[],  'PATH_SP_MASKS':[], 'PATH_LABELS':[], 'MTD_AERIAL':[]} 
        for domain in path_domains: 
            for area in os.listdir(Path(paths_data['path_aerial_'+status], domain)): 
                aerial = sorted(list(list_items(Path(paths_data['path_aerial_'+status])/domain/Path(area), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
                sen2sp = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*data.npy'))) 
                sprods = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*products.txt')))
                smasks = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*masks.npy')))
                coords = [] 
                for k in aerial: 
                    coords.append(matching_dict[k.split('/')[-1]]) 
                data['PATH_IMG'] += aerial 
                data['PATH_SP_DATA'] += sen2sp*len(aerial) 
                data['PATH_SP_DATES'] += sprods*len(aerial)
                data['PATH_SP_MASKS'] += smasks*len(aerial) 
                data['SP_COORDS'] += coords 
                if test_set == False: 
                    data['PATH_LABELS'] += sorted(list(list_items(Path(paths_data['path_labels_'+status])/domain/Path(area), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
        if config['aerial_metadata'] == True:
            data = adding_encoded_metadata(config['data']['path_metadata_aerial'], data) 

        return data 
                
    paths_data = config['data'] 
    with open(paths_data['path_sp_centroids'], 'r') as file: 
        matching_dict = json.load(file) 
    path_trainval = Path(paths_data['path_aerial_train']) 
    trainval_domains = os.listdir(path_trainval) 
    shuffle(trainval_domains) 
    idx_split = int(len(trainval_domains) * val_percent) 
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:] 
    dict_train = get_data_paths(config, train_domains, paths_data, matching_dict, test_set=False) 
    dict_val = get_data_paths(config, val_domains, paths_data, matching_dict, test_set=False) 
    path_test = Path(paths_data['path_aerial_test']) 
    test_domains = os.listdir(path_test) 
    dict_test = get_data_paths(config, test_domains, paths_data, matching_dict, test_set=True) 
    
    return dict_train, dict_val, dict_test






def adding_encoded_metadata(path_metadata_file: str, dict_paths: dict, loc_enc_size: int = 32):
    """
    For every aerial image in the dataset, get metadata, encode and add to data dict.
    """
    #### encode metadata
    def coordenc_opt(coords, enc_size=32) -> np.array:
        d = int(enc_size/2)
        d_i = np.arange(0, d / 2)
        freq = 1 / (10e7 ** (2 * d_i / d))

        x,y = coords[0]/10e7, coords[1]/10e7
        enc = np.zeros(d * 2)
        enc[0:d:2]    = np.sin(x * freq)
        enc[1:d:2]    = np.cos(x * freq)
        enc[d::2]     = np.sin(y * freq)
        enc[d + 1::2] = np.cos(y * freq)
        return list(enc)           

    def norm_alti(alti: int) -> float:
        min_alti = 0
        max_alti = 3164.9099121094  ### MAX DATASET
        return [(alti-min_alti) / (max_alti-min_alti)]        

    def format_cam(cam: str) -> np.array:
        return [[1,0] if 'UCE' in cam else [0,1]][0]

    def cyclical_enc_datetime(date: str, time: str) -> list:
        def norm(num: float) -> float:
            return (num-(-1))/(1-(-1))
        year, month, day = date.split('-')
        if year == '2018':   enc_y = [1,0,0,0]
        elif year == '2019': enc_y = [0,1,0,0]
        elif year == '2020': enc_y = [0,0,1,0]
        elif year == '2021': enc_y = [0,0,0,1]    
        sin_month = np.sin(2*np.pi*(int(month)-1/12)) ## months of year
        cos_month = np.cos(2*np.pi*(int(month)-1/12))    
        sin_day = np.sin(2*np.pi*(int(day)/31)) ## max days
        cos_day = np.cos(2*np.pi*(int(day)/31))     
        h,m=time.split('h')
        sec_day = int(h) * 3600 + int(m) * 60
        sin_time = np.sin(2*np.pi*(sec_day/86400)) ## total sec in day
        cos_time = np.cos(2*np.pi*(sec_day/86400))
        return enc_y+[norm(sin_month),norm(cos_month),norm(sin_day),norm(cos_day),norm(sin_time),norm(cos_time)] 
    
    
    with open(path_metadata_file, 'r') as f:
        metadata_dict = json.load(f)              
    for img in dict_paths['PATH_IMG']:
        curr_img     = img.split('/')[-1][:-4]
        enc_coords   = coordenc_opt([metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]], enc_size=loc_enc_size)
        enc_alti     = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
        enc_camera   = format_cam(metadata_dict[curr_img]['camera'])
        enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
        mtd_enc      = enc_coords+enc_alti+enc_camera+enc_temporal 
        dict_paths['MTD_AERIAL'].append(mtd_enc)    
        
    return dict_paths
