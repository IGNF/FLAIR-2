import torch
import torch.nn as nn
import torchvision.transforms as T

import segmentation_models_pytorch as smp  

from src.backbones.utae_model import UTAE
from src.backbones.fusion_utils import *
import torch.nn.functional as F

    
class TimeTexture_flair(nn.Module):
    """ 
     U-Tae implementation for Sentinel-2 super-patc;
     U-Net smp implementation for aerial imagery;
     Added U-Tae feat maps as attention to encoder featmaps of unet.
    """      
    
    def __init__(self,
                 config,
                 ):
        
        super(TimeTexture_flair, self).__init__()   
        

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
        

        self.fm_utae_featmap_cropped = FM_cropped(self.arch_hr.encoder_widths[0], 
                                                       list(self.arch_vhr.encoder.out_channels),
                                                       )
        self.fm_utae_featmap_collapsed = FM_collapsed(self.arch_hr.encoder_widths[0], 
                                                           list(self.arch_vhr.encoder.out_channels),
                                                           ) 
            
        if config['aerial_metadata'] == True:
            i = 512; last_spatial_dim = int([(i:=i/2) for u in range(len(self.arch_vhr.encoder.out_channels)-1)][-1])
            self.mtd_mlp = mtd_encoding_mlp(config['geo_enc_size']+13, last_spatial_dim)
        

        self.reshape_utae_output = nn.Sequential(nn.Upsample(size=(512,512), mode='nearest'),
                                                 nn.Conv2d(self.arch_hr.encoder_widths[0], config['num_classes'], 1) 
                                                )
        self.upsampling_factor = config.get('upsampling_factor', 1)

        self.center_crop_dimension = 10

    def _upsample_hr_patch(self, bspatch):
        """
        Function to upsample the high resolution patch to specified upsampling factor
        """
        if self.upsampling_factor <= 1:
            return bspatch
        
        batch_size, channels, temporal, height, width = bspatch.shape

        # Reshape to 4D (batch * temporal, channels, height, width)
        reshaped_tensor = bspatch.view(-1, channels, height, width)

        # Define the scaling factor
        scale_factor = self.upsampling_factor

        # Calculate the output size
        new_height = height * scale_factor
        new_width = width * scale_factor

        # Upsample the tensor using nn.Upsample with scale_factor
        upsample_layer = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        upsampled_tensor = upsample_layer(reshaped_tensor)

        # Reshape back to 5D
        return upsampled_tensor.view(batch_size, channels, temporal, new_height, new_width)
    
    def _crop_downsample_patch(self, patch):
        """
        Function to crop and downsample images
        """
        transform_crop = T.CenterCrop((self.upsampling_factor * self.center_crop_dimension, 
                                       self.upsampling_factor * self.center_crop_dimension))
        cropped_patch = transform_crop(patch)

        if self.upsampling_factor <= 1:
            return cropped_patch
        
        return F.interpolate(cropped_patch, size=(self.center_crop_dimension, self.center_crop_dimension), mode='bicubic', align_corners=False)

    
    def forward(self, config, bpatch, bspatch, dates, metadata):
        if self.upsampling_factor > 1:
            bspatch = self._upsample_hr_patch(bspatch)

        ### encoded feature maps and utae outputs
        _ , utae_fmaps_dec = self.arch_hr(bspatch, batch_positions=dates)  ### utae class scores and feature maps 
        unet_fmaps_enc = self.arch_vhr.encoder(bpatch)  ### unet feature maps 
        
        ### aerial metadatat encoding and adding to u-net feature maps
        if config['aerial_metadata'] == True:
            x_enc = self.mtd_mlp(metadata)
            x_enc = x_enc.unsqueeze(1).unsqueeze(-1).repeat(1,unet_fmaps_enc[-1].size()[1],1,unet_fmaps_enc[-1].size()[-1])
            unet_fmaps_enc[-1] = torch.add(unet_fmaps_enc[-1], x_enc) 
        
        ### cropped fusion module
        utae_last_fmaps_reshape_cropped = self._crop_downsample_patch(utae_fmaps_dec[-1])
        utae_last_fmaps_reshape_cropped = self.fm_utae_featmap_cropped(utae_last_fmaps_reshape_cropped, [i.size()[-1] for i in unet_fmaps_enc])       
        
        ### collapsed fusion module       
        utae_fmaps_dec_squeezed = torch.mean(utae_fmaps_dec[-1][0], dim=(-2,-1))
        utae_last_fmaps_reshape_collapsed = self.fm_utae_featmap_collapsed(utae_fmaps_dec_squeezed, [i.size()[-1] for i in unet_fmaps_enc])  ### reshape last feature map of utae to match feature maps enc. unet
        
        ### adding cropped/collasped
        utae_last_fmaps_reshape = [torch.add(i,j) for i,j in zip(utae_last_fmaps_reshape_cropped, utae_last_fmaps_reshape_collapsed)]

        ### modality dropout
        if torch.rand(1) > config['drop_utae_modality']:
            unet_utae_fmaps = [torch.add(i,j) for i,j in zip(unet_fmaps_enc, utae_last_fmaps_reshape)]  ### add utae mask to unet feats map
        else:
            unet_utae_fmaps = unet_fmaps_enc
        
        ### u-net decoding
        unet_out = self.arch_vhr.decoder(*unet_utae_fmaps)  ### unet decoder
        unet_out = self.arch_vhr.segmentation_head(unet_out) ### unet class scores 
        
        ### reshape utae output to annotation shape
        utae_out = self._crop_downsample_patch(utae_fmaps_dec[-1])
        utae_out = self.reshape_utae_output(utae_out)

        return utae_out, unet_out
    




