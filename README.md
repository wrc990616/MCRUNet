# MCRUNet
This work is proposed in "Evaluation of Breast Cancer Tumor-Infiltrating Lymphocytes on Ultrasound Images Based on A Novel Multi-cascade Residual U-shaped Network (MCRUNet)".
## preparation of environment
We have tested our code in following environment：
* torch == 1.10.0
* torchvision == 0.12.0
* python == 3.7

## Data
A total of 494 ultrasound breast images from 494 patients (223 with high TIL and 271 with low TIL) were included in this study. The images were divided into training, validation and test sets in a 3:1:1 ratio.Names and categories exist in the `.npy` file in the list folder.

`/list/`
`/list/`
`/list/`
`/list/`
`/list/`
`/list/`
## Model
![image](https://github.com/wrc990616/MCRUNet/blob/main/pic/Figure%202.jpg)

## Training

