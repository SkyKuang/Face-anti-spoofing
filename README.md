# Face-anti-spoofing
ChaLearn Face Anti-spoofing Attack Detection Challenge@CVPR2019

Usage:
environment：
    Ubuntu 16.04, GPU:GTX 1080Ti 

Dependencies：
	python == 3.6.2
	pytorch == 0.4 .1
	numpy ==1.15.2
	matplotlib == 3.0.2
	sklearn ==0.20.3
	opencv-python==3.4.3
	Pillow


Install the dependencies：

pip install -r requirements.txt

The dataset folder is shown below：


 ![dataset folder](https://github.com/SkyKuang/Face-anti-spoofing/blob/master/pic.png)

Steps：
1.	cd Face-anti-spoofing
2.	pip install -r requirement.txt  #Install the dependencies
3.	run: CUDA_VISIBLE_DEVICES=0 python train.py --root datapath 

Run：
CUDA_VISIBLE_DEVICES= 0 python train_xx.py --root /home/xxxx/datasets/CASIA-SURF




