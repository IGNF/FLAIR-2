from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import Fit_Dataset, Predict_Dataset
from .utils import *

class DataModule(LightningDataModule):
    def __init__(
        self,
        dict_train=None,
        dict_val=None,
        dict_test=None,
        num_workers=1,
        batch_size=2,
        drop_last=True,
        ref_year='2021',
        ref_date='05-15',
        pix_buff=40,
        num_classes=13,
        num_channels=5,
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes, self.num_channels = num_classes, num_channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.ref_year = ref_year
        self.ref_date = ref_date
        self.pix_buff = pix_buff

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                ref_year=self.ref_year,
                ref_date=self.ref_date,
                pix_buff=self.pix_buff,
                num_classes=self.num_classes,
            
            )

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                ref_year=self.ref_year,
                ref_date=self.ref_date,
                pix_buff=self.pix_buff,
                num_classes=self.num_classes,
            )

        elif stage == "predict":
            self.pred_dataset = Predict_Dataset(
                dict_files=self.dict_test,
                ref_year=self.ref_year,
                ref_date=self.ref_date,
                pix_buff=self.pix_buff,
                num_classes=self.num_classes,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=pad_collate_train,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=pad_collate_train,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=pad_collate_predict
        )
