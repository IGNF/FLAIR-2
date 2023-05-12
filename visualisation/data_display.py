#!/usr/bin/env python
# coding: utf-8


## Imports
import os
import re
import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import hex2color
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
import rasterio
import rasterio.plot as plot
import torch
import torchvision.transforms as T
import datetime

lut_colors = {
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
10  : '#55ff00',
11  : '#fff30d',
12  : '#e4df7c',
13  : '#3de6eb',
14  : '#ffffff',
15  : '#8ab3a0',
16  : '#6b714f',
17  : '#c5dc42',
18  : '#9999ff',
19  : '#000000'}

lut_classes = {
1   : 'building',
2   : 'pervious surface',
3   : 'impervious surface',
4   : 'bare soil',
5   : 'water',
6   : 'coniferous',
7   : 'deciduous',
8   : 'brushwood',
9   : 'vineyard',
10  : 'herbaceous vegetation',
11  : 'agricultural land',
12  : 'plowed land',
13  : 'swimming_pool',
14  : 'snow',
15  : 'clear cut',
16  : 'mixed',
17  : 'ligneous',
18  : 'greenhouse',
19  : 'other'}

## Functions

def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
         yield path.resolve().as_posix()


def remapping(lut: dict, recover='color') -> dict:
    rem = lut.copy()
    for idx in [13,14,15,16,17,18,19]: del rem[idx]
    if recover == 'color':  rem[13] = '#000000'
    elif recover == 'class':  rem[13] = 'other'
    return rem


def convert_to_color(arr_2d: np.ndarray, palette: dict = lut_colors) -> np.ndarray:
    rgb_palette = {k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()}
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in rgb_palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def display_nomenclature() -> None:   
    GS = matplotlib.gridspec.GridSpec(1,2)
    fig = plt.figure(figsize=(15,10))
    fig.patch.set_facecolor('black')

    plt.figtext(0.73,0.92, "REDUCED (BASELINE) NOMENCLATURE", ha="center", va="top", fontsize=14, color="w")
    plt.figtext(0.3, 0.92, "FULL NOMENCLATURE", ha="center", va="top", fontsize=14, color="w")

    full_nom = matplotlib.gridspec.GridSpecFromSubplotSpec(19, 1, subplot_spec=GS[0])
    for u,k in enumerate(lut_classes):
        curr_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=full_nom[u], width_ratios=[2,6])
        ax_color, ax_class = fig.add_subplot(curr_gs[0], xticks=[], yticks=[]), fig.add_subplot(curr_gs[1], xticks=[], yticks=[])
        ax_color.set_facecolor(lut_colors[k])
        ax_class.text(0.05,0.3, f'({u+1}) - '+lut_classes[k], fontsize=14, fontweight='bold')
    main_nom = matplotlib.gridspec.GridSpecFromSubplotSpec(19, 1, subplot_spec=GS[1])
    for u,k in enumerate(remapping(lut_classes, recover='class')):
        curr_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_nom[u], width_ratios=[2,6])
        ax_color, ax_class = fig.add_subplot(curr_gs[0], xticks=[], yticks=[]), fig.add_subplot(curr_gs[1], xticks=[], yticks=[])
        ax_color.set_facecolor(remapping(lut_colors, recover='color')[k])
        ax_class.text(0.05,0.3, f'({k}) - '+(remapping(lut_classes, recover='class')[k]), fontsize=14, fontweight='bold')
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('w'), spine.set_linewidth(1.5)
    plt.show()    

def display_samples(images, masks, sentinel_imgs, centroid, palette=lut_colors) -> None:
    idx= random.sample(range(0, len(images)), 1)[0]
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (10, 6)); fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.patch.set_facecolor('black')

    with rasterio.open(images[idx], 'r') as f:
        im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
    with rasterio.open(masks[idx], 'r') as f:
        mk = f.read([1])
        mk = convert_to_color(mk[0], palette=palette)
    
    sen = np.load(sentinel_imgs[idx])[20,[2,1,0],:,:]/2000
    sen_spatch = sen[:, centroid[idx][0]-int(20):centroid[idx][0]+int(20),centroid[idx][1]-int(20):centroid[idx][1]+int(20)]
    transform = T.CenterCrop(10)
    sen_aerialpatch = transform(torch.as_tensor(np.expand_dims(sen_spatch, axis=0))).numpy()
    sen = np.transpose(sen, (1,2,0))
    sen_spatch = np.transpose(sen_spatch, (1,2,0))
    sen_aerialpatch = np.transpose(sen_aerialpatch[0], (1,2,0))

    #axs = axs if isinstance(axs[], np.ndarray) else [axs]
    ax0 = axs[0][0] ; ax0.imshow(im);ax0.axis('off')
    ax1 = axs[0][1] ; ax1.imshow(mk, interpolation='nearest') ;ax1.axis('off')
    ax2 = axs[0][2] ; ax2.imshow(im); ax2.imshow(mk, interpolation='nearest', alpha=0.25); ax2.axis('off')
    ax3 = axs[1][0] ; ax3.imshow(sen);ax3.axis('off')
    ax4 = axs[1][1] ; ax4.imshow(sen_spatch);ax4.axis('off')
    ax5 = axs[1][2] ; ax5.imshow(sen_aerialpatch);ax5.axis('off')

    # Create a Rectangle patch
    rect = Rectangle((centroid[idx][1]-5.12, centroid[idx][0]-5.12), 10.24, 10.24, linewidth=1, edgecolor='r', facecolor='none')
    ax3.add_patch(rect)
    rect = Rectangle((14.88, 14.88), 10.24, 10.24, linewidth=1, edgecolor='r', facecolor='none')
    ax4.add_patch(rect)
    
    ax0.set_title('RVB Image', size=12,fontweight="bold",c='w')
    ax1.set_title('Ground Truth Mask', size=12,fontweight="bold",c='w')
    ax2.set_title('Overlay Image & Mask', size=12,fontweight="bold",c='w')
    ax3.set_title('Sentinel super area', size=12,fontweight="bold",c='w')
    ax4.set_title('Sentinel super patch', size=12,fontweight="bold",c='w')
    ax5.set_title('Sentinel over the aerial patch', size=12,fontweight="bold",c='w')   

def display_time_serie(sentinel_images, clouds_masks, sentinel_products, nb_samples, nb_dates=5):
    indices= random.sample(range(0, len(sentinel_images)), nb_samples)
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_facecolor('black')
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nb_samples, nb_dates),  # creates 2x2 grid of axes
                    axes_pad=0.25,  # pad between axes in inch.
                    )
    
    img_to_plot = []
    for u, idx in enumerate(indices):
        img_dates = read_dates(sentinel_products[idx])

        sen = np.load(sentinel_images[idx])[:,[2,1,0],:,:]/2000
        mask = np.load(clouds_masks[idx])
        dates_to_keep = filter_dates(sen, mask)
        if len(dates_to_keep)<5:
            print("Not enough cloudless dates, not filtering")
            dates = random.sample(range(0, len(sen)), nb_dates)
        else:
            sen = sen[dates_to_keep]
            img_dates = img_dates[dates_to_keep]
            dates = random.sample(range(0, len(dates_to_keep)), nb_dates)

        for d in dates:
            sen_t = np.transpose(sen[d], (1,2,0))
            img_to_plot.append((sen_t, img_dates[d]))

    for ax, (im, date) in zip(grid, img_to_plot):
        # Iterating over the grid returns the Axes.
        im = np.clip(im, 0, 1)
        ax.imshow(im, aspect='auto')
        ax.set_title(date.strftime("%d/%m/%Y"), color='whitesmoke')



                 
def display_all(images, masks) -> None:
    GS = matplotlib.gridspec.GridSpec(20,10, wspace=0.002, hspace=0.1)
    fig = plt.figure(figsize=(40,100))
    fig.patch.set_facecolor('black')
    for u,k in enumerate(images):
        ax=fig.add_subplot(GS[u], xticks=[], yticks=[])
        with rasterio.open(k, 'r') as f:
            img = f.read([1,2,3])
        rasterio.plot.show(img, ax=ax)
        ax.set_title(k.split('/')[-1][:-4], color='w')
        get_m = [i for i in masks if k.split('/')[-1].split('_')[1][:-4] in i][0]
        with rasterio.open(get_m, 'r') as f:
            msk = f.read()        
        ax.imshow(convert_to_color(msk[0], palette=lut_colors), interpolation='nearest', alpha=0.2)
    plt.show()
    
    
def display_all_with_semantic_class(images, masks: list, semantic_class: int) -> None:
    
    def convert_to_color_and_mask(arr_2d: np.ndarray, semantic_class: int, palette: dict = lut_colors) -> np.ndarray:
        rgb_palette = {k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()}
        arr_3d = np.zeros((arr_2d[0].shape[0], arr_2d[0].shape[1], 4), dtype=np.uint8)
        for c, i in rgb_palette.items():
            m = arr_2d[0] == c
            if c == semantic_class:
                g = list(i)
                g.append(150)
                u = tuple(g)
                arr_3d[m] = u
            else:
                arr_3d[m] = tuple([0,0,0,0])   
        return arr_3d  
    
    sel_imgs, sel_msks, sel_ids = [],[],[]
    for img,msk in zip(images, masks):
        with rasterio.open(msk, 'r') as f:
            data_msk = f.read()
        if semantic_class in list(set(data_msk.flatten())):
            sel_msks.append(convert_to_color_and_mask(data_msk, semantic_class, palette=lut_colors))
            with rasterio.open(img, 'r') as f:
                data_img = f.read([1,2,3])
            sel_imgs.append(data_img)
            sel_ids.append(img.split('/')[-1][:-4]) 
    if len(sel_imgs) == 0:
        print('='*50, f'      SEMANTIC CLASS: {lut_classes[semantic_class]}', '...CONTAINS NO IMAGES IN THE CURRENT DATASET!...',  '='*50, sep='\n')        
    else:
        print('='*50, f'      SEMANTIC CLASS: {lut_classes[semantic_class]}', '='*50, sep='\n')    
        GS = matplotlib.gridspec.GridSpec(int(np.ceil(len(sel_imgs)/5)),5, wspace=0.002, hspace=0.15)
        fig = plt.figure(figsize=(30,6*int(np.ceil(len(sel_imgs)/5))))
        fig.patch.set_facecolor('black')
        for u, (im,mk,na) in enumerate(zip(sel_imgs, sel_msks, sel_ids)):
            ax=fig.add_subplot(GS[u], xticks=[], yticks=[])
            ax.set_title(na, color='w')
            ax.imshow(im.swapaxes(0, 2).swapaxes(0, 1))       
            ax.imshow(mk, interpolation='nearest')
        plt.show()



def display_predictions(images, predictions, nb_samples: int, palette=lut_colors, classes=lut_classes) -> None:
    indices= random.sample(range(0, len(predictions)), nb_samples)
    fig, axs = plt.subplots(nrows = nb_samples, ncols = 2, figsize = (17, nb_samples * 8)); fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')
  
    palette = remapping(palette, recover='color')
    classes = remapping(classes, recover='class')

    for u, idx in enumerate(indices):
        rgb_image = [i for i in images if predictions[idx].split('_')[-1][:-4] in i][0]
        with rasterio.open(rgb_image, 'r') as f:
            im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(predictions[idx], 'r') as f:
            mk = f.read([1])+1
            f_classes = np.array(list(set(mk.flatten())))
            mk = convert_to_color(mk[0], palette=palette)
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0] ; ax0.imshow(im);ax0.axis('off')
        ax1 = axs[u][1] ; ax1.imshow(mk, interpolation='nearest', alpha=1); ax1.axis('off')
        if u == 0:
            ax0.set_title('RVB Image', size=16,fontweight="bold",c='w')
            ax1.set_title('Prediction', size=16,fontweight="bold",c='w')
        handles = []
        for val in f_classes:
            handles.append(mpatches.Patch(color=palette[val], label=classes[val]))
        leg = ax1.legend(handles=handles, ncol=1, bbox_to_anchor=(1.4,1.01), fontsize=12, facecolor='k') 
        for txt in leg.get_texts():
          txt.set_color('w')

def filter_dates(img, mask, clouds:bool=2, area_threshold:float=0.2, proba_threshold:int=20):
    """ Mask : array T*2*H*W
        Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
        Area_threshold : threshold on the surface covered by the clouds / snow 
        Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)

        Return array of indexes to keep
    """
    dates_to_keep = []
    
    for t in range(mask.shape[0]):
        # Filter the images with only values above 1
        c = np.count_nonzero(img[t, :, :]>1)
        if c != img[t, :, :].shape[1]*img[t, :, :].shape[2]:
            # filter the clouds / snow 
            if clouds != 2:
                cover = np.count_nonzero(mask[t, clouds, :,:]>=proba_threshold)
            else:
                cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
            cover /= mask.shape[2]*mask.shape[3]
            if cover < area_threshold:
                dates_to_keep.append(t)
    return dates_to_keep

def read_dates(txt_file: str) -> np.array:
    with open(txt_file, 'r') as f:
        products= f.read().splitlines()
    dates_arr = []
    for file in products:
        dates_arr.append(datetime.datetime(2021, int(file[15:19][:2]), int(file[15:19][2:])))
    return np.array(dates_arr)