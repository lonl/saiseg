# saiseg
 This repository is for satellite/aerial image (sai) segementation & classification.

### networks implemented
* [U-net](https://arxiv.org/abs/1505.04597)
* [Segnet](https://arxiv.org/abs/1511.00561)
* [Linknet](https://codeac29.github.io/projects/linknet/)

### next coming
* [Stacked U-net](https://arxiv.org/pdf/1804.10343.pdf)
* [Fusenet]

### usage
1. Download the ISPRS dataset
2. Modify the "data:path" in saiseg/configs/isprs_*.yml with your correct dataset path.
3. `[sudo] python train.py`

### dataLoaders implemented
* [ISPRS dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)

### todo
1. To add the visualization
2. To add other datasets implemented

### requirements
* numpy==1.12.1
* scipy==0.19.0
* pytorch >=0.4.0
* torchvision ==0.2.0
* sklearn

### related works references
* [Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
* [Related satellight/aerial image (sai) datasets](https://github.com/chrieke/awesome-satellite-imagery-competitions)
* [CVPR2018 satellight imagery analysis papers](http://openaccess.thecvf.com/CVPR2018_workshops/CVPR2018_W4.py)
* [DeepGlobe-Road-Extraction-Challenge](https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge)

### acknowlegements
This project is implemented with making references from [@meetshah1995](https://github.com/meetshah1995/pytorch-semseg).
