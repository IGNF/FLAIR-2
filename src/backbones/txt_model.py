import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torchvision.transforms as T
import segmentation_models_pytorch as smp  

from src.backbones.utae_model import UTAE


class fmap_spatial_keep_conv(nn.Module):
    """ 
    Conv. + interpolation layers to resize featmaps
    
    """
    
    def __init__(self, in_channels, out_channels_list):
        super(fmap_spatial_keep_conv, self).__init__() 
        
        conv_ = []
        for out_ch_size in out_channels_list:
            conv_.append(nn.Conv2d(in_channels=in_channels, out_channels=out_ch_size, kernel_size=1))
        self.conv_list = nn.ModuleList(conv_)

    def forward(self, mask, height_list):
        out_attn = []
        for nb, size in enumerate(height_list): 
            out_attn.append(interpolate(self.conv_list[nb](mask), size=height_list[nb], mode='bilinear'))

        return out_attn
    
    
    
class fmap_spatial_crash_mlps(nn.Module):
    """ 
    Light MLPs to summarize spatial dim of featmap

    """
    def __init__(self, in_channels, out_channels_list):
        super(fmap_spatial_crash_mlps, self).__init__()

        self.mlps = [] ## TEST 1 LAYER
        for out_ch_size in out_channels_list:
            layers = []
            m_value = int(np.abs(in_channels-out_ch_size)/3) ## 3 layers
            if in_channels < out_ch_size: 
                layers.extend([nn.Linear(in_channels, in_channels+m_value*1), nn.Dropout(0.4), nn.ReLU(inplace=True),  
                              nn.Linear(in_channels+m_value*1, in_channels+m_value*2), nn.Dropout(0.4), nn.ReLU(inplace=True), 
                              nn.Linear(in_channels+m_value*2, out_ch_size),])
                self.mlps.append(nn.Sequential(*layers).to('cuda:0'))
            elif in_channels > out_ch_size: 
                layers.extend([nn.Linear(in_channels, in_channels-m_value*1), nn.Dropout(0.4), nn.ReLU(inplace=True),  
                              nn.Linear(in_channels-m_value*1, in_channels-m_value*2), nn.Dropout(0.4), nn.ReLU(inplace=True), 
                              nn.Linear(in_channels-m_value*2, out_ch_size),])
                self.mlps.append(nn.Sequential(*layers).to('cuda:0'))  
            else:
                layers.extend([nn.Linear(in_channels, in_channels*2), nn.Dropout(0.4), nn.ReLU(inplace=True),  
                              nn.Linear(in_channels*2, in_channels*2), nn.Dropout(0.4), nn.ReLU(inplace=True), 
                              nn.Linear(in_channels*2, out_ch_size),])
                self.mlps.append(nn.Sequential(*layers).to('cuda:0'))  
        

    def forward(self, fmap, height_list):
        
        out = []
        for mlp, h in zip(self.mlps, height_list): 
            channels_ = mlp(fmap.to('cuda:0'))
            out.append(torch.unsqueeze(channels_.view(channels_.size()[0], 1).view(channels_.size()[0], 1, 1).repeat(1,h,h), 0))

        return out 
    
    

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
        
        self.fusion_mode = config["fusion"]
        
        self.arch_vhr = smp.create_model(
                                        arch="unet", 
                                        encoder_name="resnet34", 
                                        classes=config['num_classes'], 
                                        in_channels=config['num_channels_aerial'],
                                        )
       
        self.arch_hr  = UTAE(
                            input_dim=config['num_channels_sat'],
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
        
        if self.fusion_mode in ['full', 'cropped']:
            self.reshape_utae_featmap = fmap_spatial_keep_conv(self.arch_hr.encoder_widths[0], 
                                                        list(self.arch_vhr.encoder.out_channels),
                                                       )
        elif self.fusion_mode == 'summarize':
            self.reshape_utae_featmap = fmap_spatial_crash_mlps(self.arch_hr.encoder_widths[0], 
                                                        list(self.arch_vhr.encoder.out_channels),
                                                       )
        
        
            
    def forward(self, bpatch, bspatch, dates, metadata):
        
        utae_out , utae_fmaps_dec = self.arch_hr(bspatch, batch_positions=dates)  ### utae class scores and feature maps 
        unet_fmaps_enc = self.arch_vhr.encoder(bpatch)  ### unet feature maps 
        
        if self.fusion_mode == 'full':
            utae_last_fmaps_reshape = self.reshape_utae_featmap(utae_fmaps_dec[-1], [i.size()[-1] for i in unet_fmaps_enc])  ### reshape last feature map of utae to match feature maps enc. unet  

        elif self.fusion_mode == 'cropped':  
            transform = T.CenterCrop((10, 10))
            utae_last_fmaps_reshape = transform(utae_fmaps_dec[-1])    
            utae_last_fmaps_reshape = self.reshape_utae_featmap(utae_last_fmaps_reshape, [i.size()[-1] for i in unet_fmaps_enc])       
       
        elif self.fusion_mode == "summarize":
            utae_fmaps_dec_max = torch.amax(utae_fmaps_dec[-1][0], dim=(-2,-1)) ##" AMEAN"
            utae_last_fmaps_reshape = self.reshape_utae_featmap(utae_fmaps_dec_max, [i.size()[-1] for i in unet_fmaps_enc])  ### reshape last feature map of utae to match feature maps enc. unet
        
        unet_utae_fmaps = [torch.add(i,j.to(i.device)) for i,j in zip(unet_fmaps_enc, utae_last_fmaps_reshape)]  ### add utae mask to unet feats map
        
        unet_out = self.arch_vhr.decoder(*unet_utae_fmaps)  ### unet decoder
        unet_out = self.arch_vhr.segmentation_head(unet_out) ### unet class scores 
    
        return utae_out , unet_out
    




