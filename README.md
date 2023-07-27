# MCRUNet
This work is proposed in "Evaluation of Breast Cancer Tumor-Infiltrating Lymphocytes on Ultrasound Images Based on A Novel Multi-cascade Residual U-shaped Network (MCRUNet)".
## preparation of environment
We have tested our code in following environment：
* torch == 1.10.0
* torchvision == 0.12.0
* python == 3.7

## Data
A total of 494 ultrasound breast images from 494 patients (223 with high TIL and 271 with low TIL) were included in this study. The images were divided into training, validation and test sets in a 3:1:1 ratio.Names and categories exist in the `.npy` file in the list folder.

To prepare for training, we performed image preprocessing and data augmentation. 
Firstly, we manually cropped each image to remove irrelevant content while ensuring that the breast lesion region was preserved. Secondly, we resized the cropped images to 224x224 pixels without altering the aspect ratio. 

Breast ultrasound images with extraneous areas removed are shown below：

High Tils image:

![image1](https://github.com/wrc990616/MCRUNet/blob/main/pic/1H_del_black/high_tils.jpg)

Low Tils image:

![image1](https://github.com/wrc990616/MCRUNet/blob/main/pic/2L_del_black/low_tils.jpg)


## Model
![image](https://github.com/wrc990616/MCRUNet/blob/main/pic/Figure%202.jpg)

## Training

