# Face-anti-spoofing
ChaLearn Face Anti-spoofing Attack Detection Challenge@CVPR2019

Usage:
运行环境：
    Ubuntu 16.04, GPU:GTX 1080Ti 

需要库：
	python == 3.6.2
	pytorch == 0.4 .1
	numpy ==1.15.2
	matplotlib == 3.0.2
	sklearn ==0.20.3
	opencv-python==3.4.3
	Pillow


安装相关库文件：

pip install -r requirements.txt

数据集文件夹如下所示：


 ![数据集文件目录](https://github.com/SkyKuang/Face-anti-spoofing/blob/master/pic.png)

移除了预训练模型！

步骤：
1.	cd Face-anti-spoofing
2.	pip install -r requirement.txt  #安装所需库
3.	运行: CUDA_VISIBLE_DEVICES=0 python train.py --root datapath 

复现提交结果直接运行：
CUDA_VISIBLE_DEVICES= 0 python train.py --root /home/xxxx/datasets/CASIA-SURF
代码默认迭代50个epoch,代码运行结束后，将在同级目录生成output.txt文件,该文件为预测结果。



