# specularflow_highlight_deepNet


This reposititory provides data and codes used in the paper Combining Static Specular Flow and Highlight with Deep Features for Specular Surface Detection. 

## Requirement
- Python 3
- PyTorch>=1.7.1


## Dataset
- Spherical mirror dataset: [[google drive](https://drive.google.com/file/d/1yoguAVmbKL73_hr6GTQV552QihRqAOjh/view?usp=drive_link)]  
This dataset consists of 131 images of the spherical mirror, and the corresponding mask images. Of these, 104 were used as training data, and 27 were used for test data.
Images were captured in various indoor (living rooms and bedrooms) and outdoor environments (garages and gardens) by changing the distance and angle between the camera and the spherical mirror so that the mirror reflects diverse textures.



- Plastic mold dataset: [google drive]  
This dataset consists of 189 sets of a mirror-polished mold image taken in real factory
production lines, a mirror map annotating mirror surface region. Table lists the breakdown number of images for each type of mold.

<center>

|  type  |  breakdown  |
|:---- | :----: |
|  movable small case  |  18  |
|  small case  |  13  |
|  coin case  |  21  |
|  nameplate  |45|
|  movable hard case | 12|
|  hard case  |21|
|  number tag | 41|
|  coin dish  |18|

</center>

### Data structure
- Download all files from (plastic_mold_dataset[google drive] / spherical_mirror_dataset [[googl drive](https://drive.google.com/file/d/1yoguAVmbKL73_hr6GTQV552QihRqAOjh/view?usp=drive_link)])  and place them under ./data folder.

```
  data/
    ├── plastic_mold_dataset/
    │   ├── coin_case/
    │   │   ├── image/
    │   │   │   ├── img1.jpg
    │   │   │   ├── img2.jpg
    │   │   │   └── ...
    │   │   └── mask/  
    │   │       ├── img1.png
    │   │       ├── img2.png
    │   │       └── ...
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
        │   |    ├── img1.jpg
        │   |    ├── img2.jpg
        │   |    └── ...
        |   └── mask/  
        │       ├── img1.png
        │       ├── img2.png
        │       └── ...
        └── train/
            └── ...
```

## Installation
To get started with the project, follow these steps:
1. Clone the repository.
```bash
git clone https://github.com/yossi-yuto/specularflow_highlight_deepNet.git
```
2. Install the required packages using pip or conda.

## How to run
To train the model and test it on the plastic mold dataset, execute the following command:
```bash 
source pipeline_mold.sh ${GPU_NUM} ${date}
```
Also, to train the model and test it using the spherical mirror dataset, execute the following command:
```bash
source pipeline_spherical.sh ${GPU_NUM} ${date}
```
`${GPU_NUM}` is the device number of the GPU and `${date}` is the execution date argument. For instance, to run the code on September 12 using GPU 0 and the spherical mirror dataset, you can use this command:
```bash
source pipeline_spherical.sh 0 0912
```
