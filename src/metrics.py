import numpy as np
from PIL import Image
from pathlib import Path

from sklearn.metrics import confusion_matrix


def generate_miou(path_truth: str, path_pred: str) -> list:
  
    #################################################################################################
    def get_data_paths (path, filter):
        for path in Path(path).rglob(filter):
             yield path.resolve().as_posix()  
                
    def calc_miou(cm_array):
        m = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
        m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
        return m.astype(float), ious[:-1]      

    #################################################################################################
                       
    truth_images = sorted(list(get_data_paths(Path(path_truth), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
    preds_images  = sorted(list(get_data_paths(Path(path_pred), 'PRED*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
    
    if len(truth_images) != len(preds_images): 
        print('[ERROR !] mismatch number of predictions and test files.')
        return
      
    elif truth_images[0][-10:-4] != preds_images[0][-10:-4] or truth_images[-1][-10:-4] != preds_images[-1][-10:-4]: 
        print('[ERROR !] unsorted images and masks found ! Please check filenames.') 
        return
      
    else:
        patch_confusion_matrices = []

        for u in range(len(truth_images)):
            target = np.array(Image.open(truth_images[u]))-1 # -1 as model predictions start at 0 and turth at 1.
            target[target>12]=12  ### remapping masks to reduced baseline nomenclature.
            preds = np.array(Image.open(preds_images[u]))         
            patch_confusion_matrices.append(confusion_matrix(target.flatten(), preds.flatten(), labels=list(range(13))))

        sum_confmat = np.sum(patch_confusion_matrices, axis=0)
        mIou, ious = calc_miou(sum_confmat) 

        return mIou, ious
