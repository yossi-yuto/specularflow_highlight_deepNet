# specularflow_highlight_deepNet


This reposititory provides data and codes used in the paper Combining Static Specular Flow and Highlight with Deep Features for Specular Surface Detection. 

## Requirement
- Python 3
- PyTorch>=1.7.1

## Dataset
- Spherical mirror dataset: [[google drive](https://drive.google.com/file/d/1yoguAVmbKL73_hr6GTQV552QihRqAOjh/view?usp=drive_link)]  
This dataset consists of 131 images of the spherical mirror, and the corresponding mask images. Of these, 104 were used as training data, and 27 were used for test data.
Images was captured in various indoor (living rooms and bedrooms) and outdoor environments (garages and gardens) by changing the distance and angle between the camera and the spherical mirror so that the mirror reflects diverse textures.



- Plastic mold dataset: [google drive]  
We created 360 rupture scenarios for the anticipated megathrust earthquakes in the Nankai trough (Maeda et al. (2016)) to generate simulated groundmotion (image of 512 * 512) by combining possible earthquake rupture parameters, including (a) the earthquake source area and magnitude, (b) the spatial pattern of the asperity locations, and (c) the location of the rupture initiation point, as depicted in Fig. 1. The mask is extreamly sparse (see examples below) and was created based on the location of strong motion stations, [K-NET and KiK-net](https://www.kyoshin.bosai.go.jp/kyoshin/db/index_en.html) where only 1,278 pixels are 1 (observed) and other 260,866 pixels are zero (unobserved).


```
  data/
    ├── plastic_mold_dataset/
    │   ├── coin_case/
    │   │   ├── image/
    │   │   └── mask/
    │   ├── coin_tray/
    │   │   └── ...
    │   ├── hard_case/
    │   │   └── ...
    │   ├── hard_case_movable/
    │   │   └── ...
    │   ├── nameplate/
    │   │   └── ...
    │   ├── number_tag/
    │   │   └── ...
    │   ├── small_case/
    │   │   └── ...
    │   └── small_case_movable/
    │       └── ...
    │ 
    └── spherical_mirror_dataset/
        ├── test/
        │   ├── image/
        │   └── mask/
        └── train/
            └── ...
```



