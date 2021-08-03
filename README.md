# anime-WGAN-resnet-pytorch

a GAN using Wasserstein loss and resnet to generate anime pics.

一个resnet-WGAN用于生成各种二次元头像（你也可以使用别的图像数据集，用于生成图片）

@author rabbitdeng

@本项目用于深度学习中的学习交流，如有任何问题，欢迎联系我！联系方式QQ：741533684

#This project is used for learning exchanges in deep learning. If you have any questions, please feel free to contact us! Contact QQ: 741533684

#我使用了残差模块设计了了两个相对对称的残差网络，分别做生成对抗网络的的生成器与判别器，基本原理其实与DCGAN类似。在此基础上，使用了不同于Binary cross entropy loss的Wasserstein loss，
并将优化器从Adam修改为RMSprop（注意：Adam容易导致训练不稳定，且学习率不能太大。）


#I used the residual module to design two relatively symmetric residual networks, which were used as generators and discriminators to generate the confrontation network. The basic principle is actually similar to DCGAN. On this basis, Wasserstein loss, which is different from Binary cross entropy loss, is used,
And modify the optimizer from Adam to RMSprop (Note: Adam is easy to cause unstable training, and the learning rate cannot be too large.)



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

  torch-1.9.0
  
  torchvision
  
  argparse
  
  
  This model is currently still in training.I will upload a pre-trained model as soon as possible~(due to my poor graphic card)
