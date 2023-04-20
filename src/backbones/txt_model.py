import torch
import torch.nn as nn
from torch.nn.functional import interpolate

import segmentation_models_pytorch as smp  
from src.backbones.utae_model import UTAE


class apply_utae_attn(nn.Module):
    
    def __init__(self, in_channels, out_channels_list):
        super(apply_utae_attn, self).__init__() 
        
        conv_ = []
        for out_ch_size in out_channels_list:
            conv_.append(nn.Conv2d(in_channels=in_channels, out_channels=out_ch_size, kernel_size=1))
        self.conv_list = nn.ModuleList(conv_)

    def forward(self, mask, height_list):
        out_attn = []
        for nb, size in enumerate(height_list): 
            out_attn.append(interpolate(self.conv_list[nb](mask), size=height_list[nb], mode='bilinear'))

        return out_attn
    
    

class TimeTexture_flair(nn.Module):
    """ 
     U-Tae implementation for Sentinel-2 super-patc;
     U-Net smp implementation for aerial imagery;
     Added U-Tae feat maps as attention to encoder featmaps of unet.
    """      
    
    def __init__(self,
                 config,
                 use_metadata=True,
                 ):
        
        super(TimeTexture_flair, self).__init__()    
        
        self.arch_vhr = smp.create_model(
                                        arch="unet", 
                                        encoder_name="resnet34", 
                                        classes=config['num_classes'], 
                                        in_channels=config['num_channels'],
                                        )
       
        self.arch_hr  = UTAE(
                            input_dim=10,
                            encoder_widths=config["encoder_widths"], 
                            decoder_widths=config["decoder_widths"],
                            out_conv=config["out_conv"],
                            str_conv_k=config["str_conv_k"],
                            str_conv_s=config["str_conv_s"],
                            str_conv_p=config["str_conv_p"],
                            agg_mode=config["agg_mode"], 
                            encoder_norm=config["encoder_norm"],
                            n_head=config["n_head"], 
                            d_model=config["d_model"], 
                            d_k=config["d_k"],
                            encoder=False,
                            return_maps=True,
                            pad_value=config["pad_value"],
                            padding_mode=config["padding_mode"],
                            )
        
        self.reshape_utae_featmap = apply_utae_attn(self.arch_hr.encoder_widths[0], 
                                                    list(self.arch_vhr.encoder.out_channels),
                                                   )
        
        
            
    def forward(self, bpatch, bspatch, dates, metadata):
        
        utae_out , utae_fmaps_dec = self.arch_hr(bspatch, batch_positions=dates)  ### utae class scores and feature maps 
        unet_fmaps_enc = self.arch_vhr.encoder(bpatch)  ### unet feature maps 
        
        utae_last_fmaps_reshape = self.reshape_utae_featmap(utae_fmaps_dec[-1], [i.size()[-1] for i in unet_fmaps_enc]) ### reshape last feature map of utae to match feature maps enc. unet
        
        unet_fmaps_enc_attn_utae = [torch.add(i,j) for i,j in zip(unet_fmaps_enc, utae_last_fmaps_reshape)]  ### add utae mask to unet feats map
        
        unet_out = self.arch_vhr.decoder(*unet_fmaps_enc_attn_utae)  ### unet decoder
        unet_out = self.arch_vhr.segmentation_head(unet_out) ### unet class scores 
    
        return utae_out , unet_out