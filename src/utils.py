import collections.abc
import re
import yaml
import torch
import torch
from torch.nn import functional as F
from torch.utils import data

from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import get_console
from rich.style import Style
from rich.tree import Tree





def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def print_recap(config, dict_train, dict_val, dict_test):
    print('\n+'+'='*80+'+',f"{'Model name: '+config.out_model_name : ^80}", '+'+'='*80+'+', f"{'[---TASKING---]'}", sep='\n')
    for info, val in zip(["use weights", "use metadata", "use augmentation"], [config.use_weights, config.use_metadata, config.use_augmentation]): print(f"- {info:25s}: {'':3s}{val}")
    print('\n+'+'-'*80+'+', f"{'[---DATA SPLIT---]'}", sep='\n')
    for split_name, d in zip(["train", "val", "test"], [dict_train, dict_val, dict_test]): print(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
    print('\n+'+'-'*80+'+', f"{'[---HYPER-PARAMETERS---]'}", sep='\n')
    for info, val in zip(["batch size", "learning rate", "epochs", "nodes", "GPU per nodes", "accelerator", "workers"], [config.batch_size, config.learning_rate, config.num_epochs, config.num_nodes, config.gpus_per_node, config.accelerator, config.num_workers]): print(f"- {info:25s}: {'':3s}{val}")        
    print('\n+'+'-'*80+'+', '\n')

    

def print_metrics(miou, ious):
    classes = ['building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous',
               'brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
    print('\n')
    print('-'*40)
    print(' '*8, 'Model mIoU : ', round(miou, 4))
    print('-'*40)
    print ("{:<25} {:<15}".format('Class','iou'))
    print('-'*40)
    for k, v in zip(classes, ious):
        print ("{:<25} {:<15}".format(k, round(v, 5)))
    print('\n\n')    
    
    
def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)



def pad_collate_train(dict, pad_value=0):
       
    _imgs   = [i['patch'] for i in dict]    
    _sen    = [i['spatch'] for i in dict] 
    _dates  = [i['dates'] for i in dict]
    _msks   = [i['msk'] for i in dict] 
    _smsks  = [i['smsk'] for i in dict]

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
             "msk": torch.stack(_msks, dim=0),
             "smsk": torch.stack(_smsks, dim=0)
        
            }  
    return batch



def pad_collate_predict(dict, pad_value=0):
    
    _imgs   = [i['patch'] for i in dict]
    _sen    = [i['spatch'] for i in dict] 
    _dates  = [i['dates'] for i in dict]
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
             "id": _ids,
            }  
    return batch



def print_config(config: DictConfig) -> None:
    """Print content of given config using Rich library and its tree structure.
    Args: config: Config to print to console using a Rich tree.
    """
    def walk_config(tree: Tree, config: DictConfig):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                #print('HERE', group_name)
                branch = tree.add(str(group_name), style=Style(color='yellow', bold=True))
                walk_config(branch, group_option)
            elif isinstance(group_option, ListConfig):
                if not group_option:
                    #print('THERE')
                    tree.add(f'{group_name}: []', style=Style(color='yellow', bold=True))
                else:
                    #print('THA')
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='yellow', bold=True))
            else:
                if group_name == '_target_':
                    #print('THI')
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='white', italic=True, bold=True))
                else:
                    #print('THO')
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='yellow', bold=True))
    tree = Tree(
        ':deciduous_tree: Configuration Tree ',
        style=Style(color='white', bold=True, encircle=True),
        guide_style=Style(color='bright_green', bold=True),
        expanded=True,
        highlight=True,
    )
    walk_config(tree, config)
    get_console().print(tree)
