# Sepconv-superslomo
Pytorch implementation of Nikalus et. Al. “Video Frame Interpolation via Adaptive Separable Convolution,” (CVPR2017).

There are some differences between implementation of above paper and this codes.
First, this version does not use independent CUDA, cuDNN or other gpu accerlating codes but only use pytorch's inherent gpu acceration codes.
Thus this code is easy to understand or modify especially for who does not understand CUDA and cuDNN codes.

Second, this implementation use xavier initializer instead of convolution aware initialization(just for convinience, torch does not support convolution aware initialization as a module).

# Usage
Download dataset from below "Download dataset" section.

locate dataset like below directory configuration(I recommend to use folder_hierarchy_change.py).Cancel changes
`````````
|--root    
     |--train    
          |--video time line1    
               |--0000.png     
               |--0001.png    
               |--. . .    
               |--0064.png    
          |--video time line2    
               |--0000.png     
               |--0001.png    
               |--. . .     
               |--0064.png     
          |--. . .     
     |--test       
          |--video time line1     
               |--0000.png     
               |--0001.png       
               |--. . .        
               |--0064.png       
          |--video time line2      
               |--0000.png     
               |--0001.png    
               |--. .     
               |--0064.png     
          |--. . .       
`````````      
Modify config.py before run the code.


For training: python train.py


For testing: python test.py

# Download dataset
download dataset from these link:
https://www.dropbox.com/sh/duisote638etlv2/AABJw5Vygk94AWjGM4Se0Goza?dl=0

or, from this link:
http://toflow.csail.mit.edu/index.html#triplet

# Reference: 
Nikalus et. Al. “Video Frame Interpolation via Adaptive Separable Convolution,” (CVPR2017): https://arxiv.org/abs/1708.01692


HyeongminLEE/pytorch-sepconv: https://github.com/HyeongminLEE/pytorch-sepconv/tree/0078280397e1f41bf21fc15d96179cf15223c77b


sniklaus/sepconv-slomohttps: github.com/sniklaus/sepconv-slomo

deep-learning-study tistory: https://deep-learning-study.tistory.com/677

deep-learning-study tistory: https://deep-learning-study.tistory.com/678
