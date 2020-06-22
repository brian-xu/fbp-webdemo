# FBP Web Demo

Flask web-app interface wrapping [HCIILAB/SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

## Description

Images for inference can be uploaded from local files or the internet. Any faces are extracted using an OpenCV Haar-like cascade classifier,
and benchmarked using a CNN based on an AlexNet or ResNet18 architecture trained on the FBP5500 dataset.

## Notes

The cascade classifier does very poorly in detecting faces that are even slightly rotated. This could easily be remedied
by processing input images at every rotation, but that exceeds the scope of this project.

## Disclaimer

The purpose of this repository is purely educational. These models only associate faces with a number based on the
FBP-5500 dataset. There is no further meaning beyond that number, and it fails to serve as an objective measurement of
any characteristic. 

# Credits

Model architecture and weights were taken from [HCIILAB/SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).

Flask app was created with Caffe's web demo [(BVLC/Caffe)](https://github.com/BVLC/caffe/tree/master/examples/web_demo) as a reference.