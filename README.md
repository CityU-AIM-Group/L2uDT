# Semantic-oriented Labeled-to-unlabeled Distribution Translation for Image Segmentation

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/).

## Summary:

### Intoduction:
This repository is for our IEEE TMI paper ["Semantic-oriented Labeled-to-unlabeled Distribution Translation for Image Segmentation"](https://)

### Framework:
![](https://github.com/CityU-AIM-Group/L2uDT/raw/master/network.png)

## Usage:
### Requirement:
Pytorch 1.3
Python 3.6

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/L2uDT.git
cd L2uDT 
bash L2uDT.sh
```

### Data preparation:
Dataset should be put into the folder './data'. For example, if the name of dataset is CVC, then the path of dataset should be './data/CVC/', and the folder structure is as following.
```
ThresholdNet
|-data
|--CVC
|---images
|---labels
|---train.txt
|---test.txt
|---valid.txt
```
The content of 'train.txt', 'test.txt' and 'valid.txt' should be just like:
```
26.png
27.png
28.png
...
```

### Pretrained model:
You should download the pretrained model from [Google Drive](https://drive.google.com/file/d/1yeZxwV6dYHQJmj2i5x9PnB6u-rqvlkCj/view?usp=sharing), and then put it in the './model' folder for initialization. 

### Well trained model:
You could download the trained model from [Google Drive](https://drive.google.com/file), which achieves 81.464% in Dice score on the [EndoScene testing dataset](https://www.hindawi.com/journals/jhe/2017/4037190/). Put the model in directory './models'.

## Citation:
```
@article{guo2020learn,
  title={Semantic-oriented Labeled-to-unlabeled Distribution Translation for Image Segmentation},
  author={Guo, Xiaoqing and Liu, Jie, Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  year={2021},
  publisher={IEEE}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
