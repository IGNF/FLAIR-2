# Challenge FLAIR #2: textural and temporal information for semantic segmentation from multi-source optical imagery


Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset. Get in touch at ai-challenge@ign.fr


![Alt bandeau FLAIR-IGN](images/flair_bandeau.jpg?raw=true)

<div style="border-width:1px; border-style:solid; border-color:#d2db8c; padding-left: 1em; padding-right: 1em; ">
  
<h2 style="margin-top:5px;">Links</h2>


- **Datapaper :** 

- **Dataset links :** https://ignf.github.io/FLAIR/ [🛑 will made available after the FLAIR #2 challenge !]

- **Challenge page :** 

</div>






## Context & Data

The FLAIR #2 dataset is sampled countrywide and is composed of over 20 billion annotated pixels, acquired over three years and different months (spatio-temporal domains). It consists of very high resolution aerial imagery patches with 5 channels (RVB-Near Infrared-Elevation) and annotation (19 semantic classes or 13 for the baselines). High resolution Sentinel-2 1-year time series with 10 spectral band are also provided on the same areas with broader extents.
<br>

<p align="center">
  <img width="70%" src="images/flair-2-spatial.png">
  <br>
  <em>Spatial definitions of the FLAIR #2 dataset.</em>
</p>


<p align="center">
  <img width="70%" src="images/flair-2-patches.png">
  <br>
  <em>Example of input data (first three columns are from aerial imagery, fourth from Sentinel-2) and corresponding supervision masks (last column).</em>
</p>


## Baseline model 

A two-branches architecture integrating a U-Net with a pre-trained ResNet34 encoder and a U-TAE encompassing a temporal self-attention encoder is presented. The U-TAE branch aims at learning spatio-temporal embeddings from the high resolution satellite time series that are further integrated into the U-Net branch exploiting the aerial imagery. The proposed _U-T&T_ model features a fusion module to exploit and shape the U-TAE embeddings towards the U-Net branch.   

<p align="center">
  <img width="70%" src="images/flair-2-network.png">
  <br>
  <em>Example of input data (first three columns are from aerial imagery, fourth from Sentinel-2) and corresponding supervision masks (last column).</em>
</p>
