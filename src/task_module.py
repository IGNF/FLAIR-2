import torch
import torchvision.transforms as T
from torchmetrics import MeanMetric, JaccardIndex
import pytorch_lightning as pl



class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criterion,
        optimizer,
        config,
        scheduler=None,
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config=config


    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None
            self.train_metrics = JaccardIndex(task='multiclass',num_classes=self.num_classes,absent_score=1.0,reduction='elementwise_mean')
            self.val_metrics   = JaccardIndex(task='multiclass',num_classes=self.num_classes,absent_score=1.0,reduction='elementwise_mean')
            self.train_loss = MeanMetric()
            self.val_loss   = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics   = JaccardIndex(task='multiclass',num_classes=self.num_classes,absent_score=1.0,reduction='elementwise_mean')
            self.val_loss   = MeanMetric()

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)
    
    # modified
    def forward(self, config, input_patch, input_spatch, input_dates, input_mtd):
        logits = self.model(config, input_patch, input_spatch, input_dates, input_mtd)
        return logits
    
    def step(self, batch):
        
        patch      = batch['patch']
        mtd        = batch['mtd']
        spatch     = batch["spatch"]
        dates      = batch["dates"]
        targets    = batch["labels"].long()
        targets_sp = batch["slabels"].long()
        logits_utae, logits_unet = self.model(self.config, patch, spatch, dates, mtd)

        loss_unet = self.criterion[0](logits_unet, targets)
        loss_utae = self.criterion[1](logits_utae, targets_sp)       
        loss = loss_utae*self.config["w_unet_utae"][0] + loss_unet*self.config["w_unet_utae"][1]
   
        with torch.no_grad():
            preds = logits_unet.argmax(dim=1)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_step_end(self, step_output):
        loss, preds, targets = (
                                step_output["loss"].mean(),
                                step_output["preds"],
                                step_output["targets"],
                               )
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        return loss

    def training_epoch_end(self, outputs):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log(
                "train_loss",
                self.train_epoch_loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True,
                rank_zero_only=True
                )
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step_end(self, step_output):
        loss, preds, targets = (
            step_output["loss"].mean(),
            step_output["preds"],
            step_output["targets"]
        )
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)
        self.log(
            "val_miou",
            self.val_epoch_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)
        self.val_loss.reset()
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        _, logits_unet = self.forward(self.config, batch["patch"], batch['spatch'], batch['dates'], batch["mtd"])
        proba = torch.softmax(logits_unet, dim=1)
        batch["preds"] =  torch.argmax(proba, dim=1)
        return batch

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler_config = {
                                   "scheduler": self.scheduler,
                                   "interval": "epoch",
                                   "monitor": "val_loss",
                                   "frequency": 1,
                                   "strict": True,
                                   "name": "Scheduler"
                                  }
            config_ = {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
            return config_
        else: return self.optimizer
    
