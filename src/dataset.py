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
                 config,
                 augmentation_set=None
                 ):

        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_imgs_sp = np.array(dict_files["PATH_SP_DATA"])
        self.list_sp_coords = np.array(dict_files["SP_COORDS"])
        self.list_sp_products = np.array(dict_files["PATH_SP_DATES"])
        self.list_sp_masks = np.array(dict_files["PATH_SP_MASKS"])
        self.list_labels = np.array(dict_files["PATH_LABELS"])

        self.augmentation_set = augmentation_set
        self.use_metadata = config['aerial_metadata']
        if self.use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD_AERIAL"])
        
        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.average_month = config["average_month"]


    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array
        

    def read_labels(self, raster_file: str, pix_tokeep:int = 500) -> np.ndarray:
        with rasterio.open(raster_file) as src_label:
            labels = src_label.read()[0]
            labels[labels > self.num_classes] = self.num_classes
            labels = labels-1
            return labels, labels


    def read_superarea_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode='r')
        subset_sp = data[:,:,idx_centroid[0]-int(self.sat_patch_size/2):idx_centroid[0]+int(self.sat_patch_size/2),idx_centroid[1]-int(self.sat_patch_size/2):idx_centroid[1]+int(self.sat_patch_size/2)]
        return subset_sp

        
    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                              -datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days           
                             )
            dates_arr.append(datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])))
        return np.array(diff_dates), np.array(dates_arr)


    def monthly_image(self, sp_patch, sp_raw_dates):
        average_patch, average_dates  = [], []
        month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),end=sp_raw_dates[-1].strftime('%Y-%m-%d'), freq='M')
        for m in month_range:
            month_dates = list(filter(lambda i: (sp_raw_dates[i].month == m.month) and (sp_raw_dates[i].year==m.year), range(len(sp_raw_dates))))
            if len(month_dates)!=0:
                average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                average_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                                     -datetime.datetime(int(self.ref_year), int(m.month), 15)).days           
                                    )
        return np.array(average_patch), np.array(average_dates)


        
    def __len__(self):
        return len(self.list_imgs)
    


    def __getitem__(self, index):

        # aerial image
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        
        # metadata aerial images
        if self.use_metadata == True: mtd = self.list_metadata[index]
        else: mtd = []    

        # labels (+ resized to satellite resolution if asked)
        labels_file = self.list_labels[index]
        labels, s_labels = self.read_labels(raster_file=labels_file)  

        if self.augmentation_set is not None:
            sample = {"image" : img.swapaxes(0, 2).swapaxes(0, 1), "mask": labels[None,:,:].swapaxes(0, 2).swapaxes(0, 1)}
            transformed_sample = self.augmentation_set(**sample)
            img, labels = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(), transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy()[0] 

        img = img_as_float(img)

        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_file_mask = self.list_sp_masks[index]

        sp_patch = self.read_superarea_and_crop(sp_file, sp_file_coords)
        sp_dates, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask = self.read_superarea_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)

        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask)
            sp_patch = sp_patch[dates_to_keep]
            sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]

        if self.average_month:
            sp_patch, sp_dates = self.monthly_image(sp_patch, sp_raw_dates)
        
        sp_patch = img_as_float(sp_patch)

        return {"patch": torch.as_tensor(img, dtype=torch.float),
                "spatch": torch.as_tensor(sp_patch, dtype=torch.float),
                "dates": torch.as_tensor(sp_dates, dtype=torch.float),
                "labels": torch.as_tensor(labels, dtype=torch.float),
                "slabels": torch.as_tensor(s_labels, dtype=torch.float),
                "mtd": torch.as_tensor(mtd, dtype=torch.float),
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

        self.use_metadata = config['aerial_metadata']
        if self.use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD_AERIAL"])        
        
        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.average_month = config["average_month"]

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array


    def read_superarea_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode='r')
        subset_sp = data[:,:,idx_centroid[0]-int(self.sat_patch_size/2):idx_centroid[0]+int(self.sat_patch_size/2),idx_centroid[1]-int(self.sat_patch_size/2):idx_centroid[1]+int(self.sat_patch_size/2)]
        return subset_sp

        
    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                              -datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days           
                             )
            dates_arr.append(datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])))
        return np.array(diff_dates), np.array(dates_arr)


    def monthly_image(self, sp_patch, sp_raw_dates):
        average_patch, average_dates  = [], []
        month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),end=sp_raw_dates[-1].strftime('%Y-%m-%d'), freq='M')
        for m in month_range:
            month_dates = list(filter(lambda i: (sp_raw_dates[i].month == m.month) and (sp_raw_dates[i].year==m.year), range(len(sp_raw_dates))))
            if len(month_dates)!=0:
                average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                average_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                                     -datetime.datetime(int(self.ref_year), int(m.month), 15)).days           
                                    )
        return np.array(average_patch), np.array(average_dates)
        


    def __len__(self):
        return len(self.list_imgs)
    


    def __getitem__(self, index):

        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        if self.use_metadata == True: mtd = self.list_metadata[index]
        else: mtd = []    

        # Sentinel patch, dates and cloud / snow mask 
        sp_file = self.list_imgs_sp[index]
        sp_file_coords = self.list_sp_coords[index]
        sp_file_products = self.list_sp_products[index]
        sp_file_mask = self.list_sp_masks[index]

        sp_patch = self.read_superarea_and_crop(sp_file, sp_file_coords)
        sp_dates, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask = self.read_superarea_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)
      
        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask)
            sp_patch = sp_patch[dates_to_keep]
            sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]

        if self.average_month:
            sp_patch, sp_dates = self.monthly_image(sp_patch, sp_raw_dates)
        
        sp_patch = img_as_float(sp_patch)
      
        return {"patch": torch.as_tensor(img, dtype=torch.float),
                "spatch": torch.as_tensor(sp_patch, dtype=torch.float),
                "dates": torch.as_tensor(sp_dates, dtype=torch.float),
                "mtd": torch.as_tensor(mtd, dtype=torch.float),
                "id": '/'.join(image_file.split('/')[-4:])}  
