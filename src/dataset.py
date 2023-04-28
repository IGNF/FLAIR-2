import datetime
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import rasterio
from skimage import img_as_float

from src.utils_dataset import filter_dates

class Fit_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 config
                 ):

        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"])
        self.list_sp_masks = np.array(dict_files["PATH_SP_MASKS"])
        self.list_msks = np.array(dict_files["PATH_MSK"])
        
        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.mono_date = config["mono_date"]
        self.average_month = config["average_month"]


    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array
        
    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append(
                (datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                 -datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days           
            )
            dates_arr.append(datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])))
        return np.array(diff_dates), np.array(dates_arr)

    def read_sp_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file)
        subset_sp = data[:,:,idx_centroid[0]-int(self.sat_patch_size/2):idx_centroid[0]+int(self.sat_patch_size/2),idx_centroid[1]-int(self.sat_patch_size/2):idx_centroid[1]+int(self.sat_patch_size/2)]
        return data, subset_sp
        
    def read_msk(self, raster_file: str, pix_tokeep:int = 500) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            msk = src_msk.read()[0]
            msk[msk > self.num_classes] = self.num_classes
            msk = msk-1
            
            # array = np.stack([array == i for i in range(self.num_classes)], axis=0) #needed for cross entropy in some vroisn of torch

            # Resize and crop the labels for the sentinel part
            to_remove = (msk.shape[0] - pix_tokeep)//2
            subset_msk = msk[to_remove:to_remove+pix_tokeep,to_remove:to_remove+pix_tokeep]
            resized_msk = T.Resize(10, interpolation=InterpolationMode.NEAREST)(torch.as_tensor(subset_msk, dtype=torch.int).unsqueeze(0))
            resized_msk = torch.squeeze(resized_msk)
            
            return msk, resized_msk
        
    def average_images(self, sp_patch, sp_raw_dates):
        average_patch = []
        average_dates = []
        month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),end=sp_raw_dates[-1].strftime('%Y-%m-%d'), freq='M')
        for m in month_range:
            month_dates = list(filter(lambda i: (sp_raw_dates[i].month == m.month) and (sp_raw_dates[i].year==m.year), range(len(sp_raw_dates))))
            if len(month_dates)!=0:
                average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                average_dates.append(
                    (datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                        -datetime.datetime(int(self.ref_year), int(m.month), 15)).days           
                )
        return np.array(average_patch), np.array(average_dates)

        
    def __len__(self):
        return len(self.list_imgs)
    

    def __getitem__(self, index):

        # Ortho image
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        # Labels at ortho and sentinel resolution
        mask_file = self.list_msks[index]
        msk, smsk = self.read_msk(raster_file=mask_file)  

        # Sentinel patch, dates and cloud / snow mask 
        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_file_mask = self.list_sp_masks[index]

        _, sp_patch = self.read_sp_and_crop(sp_file, sp_file_coords)
        sp_dates, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask_zone, sp_mask = self.read_sp_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)
        sp_mask_zone = sp_mask_zone.astype(int)

        
        
        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask_zone)
            #print("nb dates before : ",  len(sp_dates))
            sp_patch = sp_patch[dates_to_keep]
            sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]
            #print("nb dates after : ", len(sp_dates))

        if self.average_month:
            sp_patch, sp_dates = self.average_images(sp_patch, sp_raw_dates)

        if self.mono_date:
            closest_date_reference = [np.argmin(np.abs(sp_dates))]
            #print("Keeping only one date : {} days away from the reference date".format(abs(sp_dates[closest_date_reference[0]])))
            sp_patch = sp_patch[closest_date_reference]
            sp_dates = sp_dates[closest_date_reference]
        
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
                 config
                 ):
        
        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"])
        self.list_sp_masks = np.array(dict_files["PATH_SP_MASKS"])
        
        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.mono_date = config["mono_date"]
        self.average_month = config["average_month"]

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array
        
    def read_sp_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file)
        subset_sp = data[:,:,idx_centroid[0]-int(self.sat_patch_size/2):idx_centroid[0]+int(self.sat_patch_size/2),idx_centroid[1]-int(self.sat_patch_size/2):idx_centroid[1]+int(self.sat_patch_size/2)]
        return data, subset_sp        
        
    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append(
                (datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                 -datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days           
            )
            dates_arr.append(datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])))
        return np.array(diff_dates), np.array(dates_arr)
    
    def average_images(self, sp_patch, sp_raw_dates):
        average_patch = []
        average_dates = []
        month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),end=sp_raw_dates[-1].strftime('%Y-%m-%d'), freq='M')
        for m in month_range:
            month_dates = list(filter(lambda i: (sp_raw_dates[i].month == m.month) and (sp_raw_dates[i].year==m.year), range(len(sp_raw_dates))))
            if len(month_dates)!=0:
                average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                average_dates.append(
                    (datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                        -datetime.datetime(int(self.ref_year), int(m.month), 15)).days           
                )
        return np.array(average_patch), np.array(average_dates)
        
    def __len__(self):
        return len(self.list_imgs)
    

    def __getitem__(self, index):

        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        # Sentinel patch, dates and cloud / snow mask 
        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_file_mask = self.list_sp_masks[index]

        _, sp_patch = self.read_sp_and_crop(sp_file, sp_file_coords)
        sp_dates, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask_zone, sp_mask = self.read_sp_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)
        sp_mask_zone = sp_mask_zone.astype(int)
      
        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask_zone)
            #print("nb dates before : ",  len(sp_dates))
            sp_patch = sp_patch[dates_to_keep]
            sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]
            #print("nb dates after : ", len(sp_dates))

        if self.average_month:
            sp_patch, sp_dates = self.average_images(sp_patch, sp_raw_dates)
            

        if self.mono_date:
            closest_date_reference = [np.argmin(np.abs(sp_dates))]
            #print("Keeping only one date : {} days away from the reference date".format(abs(sp_dates[closest_date_reference[0]])))
            sp_patch = sp_patch[closest_date_reference]
            sp_dates = sp_dates[closest_date_reference]
        
        sp_patch = img_as_float(sp_patch)
      
        return {"patch": torch.as_tensor(img, dtype=torch.float),
                "spatch": torch.as_tensor(sp_patch, dtype=torch.float),
                "dates": torch.as_tensor(sp_dates, dtype=torch.float),
                "id": '/'.join(image_file.split('/')[-4:])}  