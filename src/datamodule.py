from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import Fit_Dataset, Predict_Dataset
from .utils_dataset import *

class DataModule(LightningDataModule):
    def __init__(
        self,
        dict_train=None,
        dict_val=None,
        dict_test=None,
        config=None,
        drop_last=True,
        augmentation_set=None,
        
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.config = config
        self.augmentation_set = augmentation_set


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                config=self.config,
                augmentation_set=self.augmentation_set,
            )

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                config=self.config
            )

        elif stage == "predict":
            self.pred_dataset = Predict_Dataset(
                dict_files=self.dict_test,
                config=self.config
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            drop_last=self.drop_last,
            collate_fn=pad_collate_train,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            drop_last=self.drop_last,
            collate_fn=pad_collate_train,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=self.config['batch_size_inference'], 
            shuffle=False,
            num_workers=self.config["num_workers"],
            drop_last=self.drop_last,
            collate_fn=pad_collate_predict
        )
