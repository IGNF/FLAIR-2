from omegaconf import DictConfig, ListConfig
from rich import get_console
from rich.style import Style
from rich.tree import Tree
from pytorch_lightning.utilities import rank_zero_only 
from datetime import timedelta

@rank_zero_only
def print_recap(config, dict_train, dict_val, dict_test):
    print('\n+'+'='*80+'+',f"{'Model name: '+config.out_model_name : ^80}", '+'+'='*80+'+', f"{'[---TASKING---]'}", sep='\n')
    for info, val in zip(["use metadata", "use augmentation"], [config.use_metadata, config.use_augmentation]): print(f"- {info:25s}: {'':3s}{val}")
    print('\n+'+'-'*80+'+', f"{'[---DATA SPLIT---]'}", sep='\n')
    for split_name, d in zip(["train", "val", "test"], [dict_train, dict_val, dict_test]): print(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
    print('\n+'+'-'*80+'+', f"{'[---HYPER-PARAMETERS---]'}", sep='\n')
    for info, val in zip(["batch size", "learning rate", "epochs", "nodes", "GPU per nodes", "accelerator", "workers"], [config.batch_size, config.learning_rate, config.num_epochs, config.num_nodes, config.gpus_per_node, config.accelerator, config.num_workers]): print(f"- {info:25s}: {'':3s}{val}")        
    print('\n+'+'-'*80+'+', '\n')
       
@rank_zero_only
def print_inference_time(tt, config): 
    tt = tt * (config['num_nodes']*config['gpus_per_node'])
    print('','','#'*80,' '*28+'--- INFERENCE TIME ---', sep='\n')
    print('- nodes: ', config['num_nodes'])
    print('- gpus per nodes: ', config['gpus_per_node'])
    print('[MAX FOR VALID MODEL] : 0:25:00 HH:MM:SS',
          f'[CURRENT]             : {str(timedelta(seconds=tt))} HH:MM:SS','',sep='\n') 
    if tt > 1500: print('[X] INFERENCE TOO LONG')
    else: print('[V] INFERENCE TIME BELOW MAX !', '\n\n')
    print('#'*80, '\n\n')
    
    
@rank_zero_only
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

@rank_zero_only
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
