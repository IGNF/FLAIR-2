from pathlib import Path 
import json
from random import shuffle
import os



def load_data (config: dict, val_percent=0.8): 
    """ Returns dicts (train/val/test) with 5 keys: 
    - PATH_IMG : aerial image (path, str) 
    - PATH_SP_DATA : satellite image (path, str) 
    - PATH_SP_DATES : satellite product names (path, str) 
    - SP_COORDS : centroid coordinate of patch in superpatch (list, e.g., [56,85]) 
    - PATH_MSK : labels (path, str) """ 
    def get_data_paths(path_domains: str, paths_data: dict, matching_dict: dict, test_set: bool) -> dict: 
        #### return data paths 
        def list_items(path, filter): 
            for path in Path(path).rglob(filter): 
                yield path.resolve().as_posix() 
        status = ['train' if test_set == False else 'test'][0] 
        ## data paths dict
        data = {'PATH_IMG':[], 'PATH_SP_DATA':[], 'SP_COORDS':[], 'PATH_SP_DATES':[], 'PATH_MSK':[]} 
        for domain in path_domains: 
            for area in os.listdir(Path(paths_data['path_aerial_'+status], domain)): 
                aerial = sorted(list(list_items(Path(paths_data['path_aerial_'+status])/domain/Path(area), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
                sen2sp = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*data.npy'))) 
                sprods = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*products.txt'))) 
                coords = [] 
                for k in aerial: 
                    coords.append(matching_dict[k.split('/')[-1]]) 
                data['PATH_IMG'] += aerial 
                data['PATH_SP_DATA'] += sen2sp*len(aerial) 
                data['PATH_SP_DATES'] += sprods*len(aerial) 
                data['SP_COORDS'] += coords 
                if test_set == False: 
                    data['PATH_MSK'] += sorted(list(list_items(Path(paths_data['path_labels_'+status])/domain/Path(area), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
        return data 
                
    paths_data = config['data'] 
    with open(paths_data['path_sp_centroids'], 'r') as file: 
        matching_dict = json.load(file) 
    path_trainval = Path(paths_data['path_labels_train']) 
    trainval_domains = os.listdir(path_trainval) 
    shuffle(trainval_domains) 
    idx_split = int(len(trainval_domains) * val_percent) 
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:] 
    dict_train = get_data_paths(train_domains, paths_data, matching_dict, test_set=False) 
    dict_val = get_data_paths(val_domains, paths_data, matching_dict, test_set=False) 
    path_test = Path(paths_data['path_labels_test']) 
    test_domains = os.listdir(path_test) 
    #print(test_domains) 
    dict_test = get_data_paths(test_domains, paths_data, matching_dict, test_set=True) 
    
    
    print('DATALOADER TRAIN -->  IMGS: ', len(dict_train['PATH_IMG']), '  SP-IMGS: ', len(dict_train['PATH_SP_DATA']))
    print('DATALOADER VAL   -->  IMGS: ', len(dict_val['PATH_IMG']), '  SP-IMGS: ', len(dict_val['PATH_SP_DATA']))
    print('DATALOADER TEST  -->  IMGS: ', len(dict_test['PATH_IMG']), '  SP-IMGS: ', len(dict_test['PATH_SP_DATA']))    
    
    return dict_train, dict_val, dict_test