# Camera-Calibration-Auto-Calib
This project implements the robust camera cali-
bration technique proposed by Z. Zhang in ”A Flexible New
Technique for Camera Calibration”. The method, which
requires at least two images at different orientations, is tested
on a dataset of 13 images captured using a Google Pixel XL
phone. The camera parameters are estimated in two stages:
first, computing homographies for each image to obtain initial
camera intrinsics through a closed-form solution, then calculating
extrinsic parameters for each image. Finally, these parameters,
along with radial distortion coefficients, are refined using the
Levenberg-Marquardt optimization algorithm to minimize re-
projection error. The pipeline of the entire implementation can be found in the report below.

Report Link: - https://drive.google.com/file/d/13HsdwWuA8Zq7M5G5mxl2rAoO3xD7zAqP/view?usp=sharing
