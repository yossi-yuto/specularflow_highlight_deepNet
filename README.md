# specularflow_highlight_deepNet


This reposititory provides data and codes used in the paper Position-Dependent Partial Convolutions for Supervised Spatial Interpolation in [ACML2022](https://proceedings.mlr.press/v189/hachiya23a/hachiya23a.pdf) and [MLWA (submitted)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4311229). 

## Requirement
- Python 3
- PyTorch>=1.7.1

## Dataset
- Toy dataset  
To reproduce the problem of translation invariance in spatial interpolation, we designed an image of 256 * 256 to contain a rectangular object with different widths depending on its vertical position---narrow in the upper-half and wide in the lower-half, and the fixed mask containing several windows (see an example below) of the same width vertically where the rectangular objects in upper and lower-half look exactly same.
  
- Simulated ground motion datasets
We created 360 rupture scenarios for the anticipated megathrust earthquakes in the Nankai trough (Maeda et al. (2016)) to generate simulated groundmotion (image of 512 * 512) by combining possible earthquake rupture parameters, including (a) the earthquake source area and magnitude, (b) the spatial pattern of the asperity locations, and (c) the location of the rupture initiation point, as depicted in Fig. 1. The mask is extreamly sparse (see examples below) and was created based on the location of strong motion stations, [K-NET and KiK-net](https://www.kyoshin.bosai.go.jp/kyoshin/db/index_en.html) where only 1,278 pixels are 1 (observed) and other 260,866 pixels are zero (unobserved).
There are three datasets as follows:
    - All---all 360 scenarios using 10-fold cross-validation
    - East---150 scenarios with east rapture initiation points, h04, h05, h09 and h10 (see Fig.1 (c)) using 5-fold cross-validation
    - West---140 scenarios of west rapture initiation points, h01, h02, h06 and h07 (see Fig.1 (c)) using 5-fold cross-validation
<p align="center">
<img src="images/asperity-fig3_en.png"  width="60%" height="60%">  
</p>
<p align="center">
Fig. 1 Variability of the possible earthquake rupture parameters for the anticipated megathrust earthquakes in the Nankai trough.
</p>
All datasets (one toy and three for ground motion) consists of train, validation and test.  

### Preparation
- Download all files from [googl drive](https://drive.google.com/drive/folders/1A8SPMM7OBdZfD8x2EeQ3G-pYih4hfINJ) and place them under ./data folder.
- Visualize images and masks in each dataset using the following command:
~~~
python visualize_data.py [-data] [-mode] [-cv] [-start_ind] [-display_num]
~~~

- data
    - String : name of data, \'quake-all\', \'quake-east\', \'quake-west\', or \'toy\' (default=quake-all)
- mode
    - String : name of mode, \'train\', \'test\', or \'valid\' (default=train)'
- cv
    - Int : number of cross validation, 0-9 (default=0)
- start_ind
    - Int : start index (default=0)
- display_num
    - Int : number of images displayed (default=5)

### Examples of visualization
~~~bash
python visualize_data.py -data toy -mode train -start_ind 75 -display_num 1
~~~
<p align="center">
<img src="images/toy_train_75.png"  width="60%" height="60%">  
</p>

~~~bash
python visualize_data.py -data toy -mode train -start_ind 140 -display_num 1
~~~
<p align="center">
<img src="images/toy_train_140.png"  width="60%" height="60%">  
</p>

~~~bash
python visualize_data.py -data quake-all -mode train -start_ind 0 -display_num 2
~~~
<p align="center">
<img src="images/quake_all_train_0.png"  width="60%" height="60%">  
<img src="images/quake_all_train_1.png"  width="60%" height="60%">  
</p>

## Codes of spatial interpolation
coming soon

## Codes of deep inpainting
coming soon

## Citation
```
@inproceedings{acml:hachiya:2022,
  title={Position-dependent partial convolutions for supervised spatial interpolation},
  author={Hirotaka Hachiya, Kotaro Nagayoshi, Asako Iwaki, Takahiro Maeda, Naonori Ueda, Hiroyuki Fujiwara},
  booktitle={Proceedings of the 14th Asian Conference on Machine Learning (ACML)},
  year={2022}
}

@article{SRL:Maeda:2016,
  title = {Seismic-Hazard Analysis of Long-Period Ground Motion of Megathrust Earthquakes in the Nankai Trough Based on 3D Finite-Difference Simulation},
  author = {Takahiro Maeda and Asako Iwaki and Nobuyuki Morikawa and Shin Aoi and Hiroyuki Fujiwara},
  journal = {SEISMOLOGICAL RESEARCH LETTERS},
  month = {11},
  number = {6},
  pages = {1265--1273},
  publisher = {Seismological Society of America},
  doi = {https://doi.org/10.1785/0220160093},
  volume = {87},
  year = {2016}  
}
