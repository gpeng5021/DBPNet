This project is the official code of the paper [DBPNet: A Dual-Branch Pyramid Network for Document Super-Resolution].

## Folder structure

The project folder for this section should be placed in the following structure:

```
DBPNet
├── RSM	#Folder of RSM code
├── data
├── dataset	#training dataset and testing dataset
│   ├── test 
│   ├── train 
│   ├── train300
│   ├── results
│   ├── trained_models
│   ├── total_crop_no_border
├── loss	
├── model	
├── models	
├── utils	
├── calc_psnr_ssim_text330.m	#MATLAB code for calculating PSNR and SSIM
├── create_data.py
├── crop_pic.py	
├── data_argument.py
├── make_border.py
├── readme.md
├── requirements.txt
├── test_dbpnet_x2.py	
├── test_dbpnet_x2.py	
├── train_x2_add.py 	
├── train_x4_add.py 	  
```

## Requirements

Required environment

1. Python 3.6.13
2. torch 1.10.0
3. numpy 1.19.2
4. cuda 11.3

Install the environment with following commands.

```
conda create -n dbpnet_env python==3.6.13
conda activate dbpnet_env
pip install -r requirements.txt
```

## prepare data

1.Download the original Text330 datastet from https://github.com/t2mhanh/LapSRN_TextImages_git

2.Download the cropped Text330 datastet from https://pan.baidu.com/s/1-EN1nLSX6nX0OifrHMEcEw with code ys8l. This is our
training and testing datatset. Please place the downloaded dataset folder in the path of this project [DBPNet / dataset]
.

(1)“train” folder

The total_crop folder contains 300 cropped ground truth images in Text330 datastet.

The total_crop_x2 folder contains images processed with ×4 downsampling.

The total_crop_x4 folder contains images processed with ×2 downsampling.

The segment folder contains images processed with RSM segmentation.

In these images, seg_img_hr and seg_text_hr are images of respectively plain images and plain texts with RSM
segmentation from total_crop folder.

Seg_img_×2 and seg_text_×2 are images of respectively plain images and plain texts of seg_img_hr and seg_text_hr with ×4
downsampling.

Seg_img_×4 and seg_text_×4 are images of respectively plain images and plain texts of seg_img_hr and seg_text_hr with ×2
downsampling.

(2) “test” folder

The total_crop folder contains 30 ground truth images in Text330 dataset.

The total_crop_x2 folder contains images from total_crop folder processed with ×4 downsampling.

The total_crop_x4 folder contains images from total_crop folder processed with ×2 downsampling.

The segment folder contains images processed with RSM segmentation.

In these images, seg_img_hr and seg_text_hr are images of respectively plain images and plain texts with RSM
segmentation from total_crop folder.

Seg_img_×2 and seg_text_×2 are images of respectively plain images and plain texts of seg_img_hr and seg_text_hr with
RSM.

Seg_img_×4 and seg_text_×4 are images of respectively plain images and plain texts of seg_img_hr and seg_text_hr with
RSM.

method of making training and testing datasets

(1)Crop ground truth images and enhance the data of the cropped images

```
python crop_pic.py  # obtain randomly-cropped images from ground truth,生成的图片保存在train300_crop
python data_argument.py # enhance data of the cropped images，生成的图片分别保存在train300_crop_ro,train300_crop_ho,train300_crop_ve,train300_crop_hv
mkdir total_crop_no_border
将train300_crop,train300_crop_ro,train300_crop_ho,train300_crop_ve,train300_crop_hv文件夹下的全部图片放到total_crop_no_border文件夹下
python make_border.py # obtain cropped images of the same size
python create_data.py   # downsample the cropped pictures at scale factors
```

(2)Obtain segmented images with RSM

```
cd data/train/ 
mkdir segment
cd RSM
python create_seg_data_origin.py
```

## train

1.Run following commands to train

```
python train_x2_add.py
python train_x4_add.py
NOTE：Please find trained models in DBPNet/data/trained_models/.
```

## test

1.The models folder contains trained models 2.Run following commands to test and verify:

```
python test_dbpnet_x2.py  
python test_dbpnet_x4.py  
Run the code calc_psnr_ssim_text330.m from https://github.com/t2mhanh/LapSRN_TextImages_git  with MATLAB  to verify.
NOTE：Please find testing results in DBPNet/data/results/.
```

