# Sepconv-superslomo
Pytorch implementation of Nikalus et. Al. “Video Frame Interpolation via Adaptive Separable Convolution,” (CVPR2017).

There are some differences between implementation of above paper and this codes.
First, this code does not use CUDA, cuDNN or other gpu accerlating methods but only use torch's inherent gpu acceration.
Thus this code does not require CUDA or cuDNN toolkit, and easy to understand or modify especially for who does not understand CUDA and cuDNN codes.

Second, this code use xavier initializer instead of convolution aware initialization.

# Usage
Modify config.py before run the code.


For training: python train.py


For testing: python test.py

# Download dataset
download dataset from these link:
https://www.dropbox.com/sh/duisote638etlv2/AABJw5Vygk94AWjGM4Se0Goza?dl=0

or, from this link:
http://toflow.csail.mit.edu/index.html#triplet

# Reference: 
https://github.com/HyeongminLEE/pytorch-sepconv/tree/0078280397e1f41bf21fc15d96179cf15223c77b


https://github.com/sniklaus/sepconv-slomo
