import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import rasterio
from skimage import img_as_float



class Fit_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 ref_year = '2021',
                 ref_date = '05-15',
                 pix_buff = 40,
                 num_classes=13, 
                 ):

        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"])
        self.list_msks = np.array(dict_files["PATH_MSK"])
        
        self.ref_year = ref_year
        self.ref_date = ref_date
        self.pix_buff = pix_buff
        self.num_classes = num_classes


    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array
        
    def read_dates(self, txt_file: str, ref_year: str, ref_date: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        dates_arr = []
        for file in products:
            dates_arr.append(
                (datetime(int(ref_year), int(ref_date.split('-')[0]), int(ref_date.split('-')[1])) 
                 -datetime(int(ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days           
            )
        return np.array(dates_arr)

    def read_sp_and_crop(self, numpy_file: str, idx_centroid: list, pix_buff: int) -> np.ndarray:
        data = np.load(numpy_file)
        subset_sp = data[:,:,idx_centroid[0]-int(pix_buff/2):idx_centroid[0]+int(pix_buff/2),idx_centroid[1]-int(pix_buff/2):idx_centroid[1]+int(pix_buff/2)]
        return subset_sp
        
    def read_msk(self, raster_file: str, pix_tokeep:int = 500) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            data = src_msk.read()[0]
            data[data > self.num_classes] = self.num_classes
            data = data-1
            
            # array = np.stack([array == i for i in range(self.num_classes)], axis=0) #needed for cross entropy in some vroisn of torch

            # Resize and crop the labels for the sentinel part
            to_remove = (data.shape[0] - pix_tokeep)//2
            subset_msk = data[to_remove:to_remove+pix_tokeep,to_remove:to_remove+pix_tokeep]
            resized_msk = T.Resize(10, interpolation=InterpolationMode.NEAREST)(torch.as_tensor(subset_msk, dtype=torch.int).unsqueeze(0))
            final_msk = torch.squeeze(resized_msk)
            
            return data, final_msk
      
        
    def __len__(self):
        return len(self.list_imgs)
    

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)        
        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_patch = self.read_sp_and_crop(sp_file, sp_file_coords, self.pix_buff)
        sp_dates = self.read_dates(sp_file_products, self.ref_year, self.ref_date)
        
        mask_file = self.list_msks[index]
        msk, smsk = self.read_msk(raster_file=mask_file)
        
        img = img_as_float(img)
        sp_patch = img_as_float(sp_patch)

        return {"patch": torch.as_tensor(img, dtype=torch.float),
                "spatch": torch.as_tensor(sp_patch, dtype=torch.float),
                "dates": torch.as_tensor(sp_dates, dtype=torch.float),
                "msk": torch.as_tensor(msk, dtype=torch.float),
                "smsk": torch.as_tensor(smsk, dtype=torch.float)
                }           



class Predict_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 ref_year = '2021',
                 ref_date = '05-15',
                 pix_buff = 20,
                 num_classes=13,
                 ):
        
        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"]) 
        
        self.ref_year = ref_year
        self.ref_date = ref_date
        self.pix_buff = pix_buff
        self.num_classes = num_classes

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array
        
    def read_sp_and_crop(self, numpy_file: str, idx_centroid: list, pix_buff: int) -> np.ndarray:
        data = np.load(numpy_file)
        subset_sp = data[:,:,idx_centroid[0]-int(pix_buff/2):idx_centroid[0]+int(pix_buff/2),idx_centroid[1]-int(pix_buff/2):idx_centroid[1]+int(pix_buff/2)]
        return subset_sp        
        
    def read_dates(self, txt_file: str, ref_year: str, ref_date: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        dates_arr = []
        for file in products:
            dates_arr.append(
                (datetime(int(ref_year), int(ref_date.split('-')[0]), int(ref_date.split('-')[1]))  #### REF. YEAR HARDCODED
                 -datetime(int(ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days  #### REF. YEAR HARDCODED            
            )
        return np.array(dates_arr)
        
    def __len__(self):
        return len(self.list_imgs)
    

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        
        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_patch = self.read_sp_and_crop(sp_file, sp_file_coords, self.pix_buff)
        sp_dates = self.read_dates(sp_file_products, self.ref_year, self.ref_date)
        
        img = img_as_float(img)
        sp_patch = img_as_float(sp_patch)

        return {"patch": torch.as_tensor(img, dtype=torch.float),
                "spatch": torch.as_tensor(sp_patch, dtype=torch.float),
                "dates": torch.as_tensor(sp_dates, dtype=torch.float),
                "id": '/'.join(image_file.split('/')[-4:])}  