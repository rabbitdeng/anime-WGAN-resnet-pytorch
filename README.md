# anime-WGAN-resnet-pytorch


![fake_samples_epoch_300](https://user-images.githubusercontent.com/88369122/132981320-f8d18028-4f95-47dc-a2f5-3dc7eb755d00.png)
#以上是在1000个图片上训练300epoch后的效果，用时大概一小时（RTX3070一块）
a GAN using Wasserstein loss and resnet to generate anime pics.

一个resnet-WGAN用于生成各种二次元头像（你也可以使用别的图像数据集，用于生成图片）

@author rabbitdeng

@本项目用于深度学习中的学习交流，如有任何问题，欢迎在Issues中提问！

#This project is used for learning exchanges in deep learning. If you have any questions, please feel free to contact us! Contact QQ: 741533684

#我使用了残差模块设计了了两个相对对称的残差网络，分别做生成对抗网络的的生成器与判别器，基本原理其实与DCGAN类似。在此基础上，使用了不同于Binary cross entropy loss的Wasserstein loss，
并将优化器从Adam修改为RMSprop（注意：Adam容易导致训练不稳定，且学习率不能太大。）


#I used the residual module to design two relatively symmetric residual networks, which were used as generators and discriminators to generate the confrontation network. The basic principle is actually similar to DCGAN. On this basis, Wasserstein loss, which is different from Binary cross entropy loss, is used,
And modify the optimizer from Adam to RMSprop (Note: Adam is easy to cause unstable training, and the learning rate cannot be too large.)


#2021/9/2

#突然发现看到的人还挺多的，这个项目的超参数并没有设置到比较优秀的水平，也很容易出现model collapse，建议大伙调低学习率，生成64*64像素图片试试。比较惭愧😂，我也是GAN的初学者，由于残差网络做GAN训练难度相对很高，我也在学习训练技巧，如果大家不一定要使用RESNET的话，不妨尝试我的另一个SAGAN的项目。日后的学习中，我会做好一个

The file directory is as follows:
--------------------------------

|

|
---data

|

|
---resnetimg

|

|
---losses.py

|

|
---model.py

|

|
---train.py






The following library files are required：
-----------------------------

  torch-1.9.0
  
  torchvision
  
  argparse
  
  
  This model is currently still in training.I will upload a pre-trained model as soon as possible~(due to my poor graphic card)
  
Train your own model(训练你自己的模型):
---------------------

#已将数据集上传至百度网盘，连接如下：

#链接：https://pan.baidu.com/s/1FWSmO5ZClyDy7YIlFwY7pw

#提取码：wwdy


#download the dataset at googledrive:
https://drive.google.com/file/d/1fMJrg2KH0S00PO2SK8in3BArU8MbTe1J/view?usp=sharing

