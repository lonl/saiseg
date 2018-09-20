# saiseg
 This repository is for satellite/aerial image (sai) segementation & classification.

## Networks implemented
* [U-net](https://arxiv.org/abs/1505.04597)
* [Segnet](https://arxiv.org/abs/1511.00561)
* [Linknet](https://codeac29.github.io/projects/linknet/)

## Next coming
* [Stacked U-net](https://arxiv.org/pdf/1804.10343.pdf)
* [Fusenet]

## Usage
1. Download the ISPRS dataset
2. Modify the "data:path" in saiseg/configs/isprs_*.yml with your correct dataset path.
3. `[sudo] python train.py`

## DataLoaders implemented
* [ISPRS dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)

## TODO
1. To add the visualization
2. To add other datasets implemented

## Requirements
* numpy==1.12.1
* scipy==0.19.0
* pytorch >=0.4.0
* torchvision ==0.2.0
* sklearn

## Related works references
* [Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
* [Related satellight/aerial image (sai) datasets](https://github.com/chrieke/awesome-satellite-imagery-competitions)
* [CVPR2018 satellight imagery analysis papers](http://openaccess.thecvf.com/CVPR2018_workshops/CVPR2018_W4.py)
* [DeepGlobe-Road-Extraction-Challenge](https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge)

## Acknowlegements
This project is implemented with making references from [@meetshah1995](https://github.com/meetshah1995/pytorch-semseg).
