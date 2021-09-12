import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from tqdm import tqdm
from model import netD,netG
from losses import Hinge
import os
#已将数据集上传至百度网盘，连接如下：
#链接：https://pan.baidu.com/s/1FWSmO5ZClyDy7YIlFwY7pw
#提取码：wwdy
#
#
#
#

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lrd', type=float, default=5e-5, help="Discriminator's learning rate, default=0.00005")  #Discriminator's learning rate
parser.add_argument('--lrg', type=float, default=5e-5, help="Generator's learning rate, default=0.00005")  #Generator's learning rate
parser.add_argument('--data_path', default='data/', help='folder to train data')#将数据集放在此处
parser.add_argument('--outf', default='resnetimg/', help='folder to output images and model checkpoints')#输出生成图片以及保存模型的位置
opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)
netG = netG().to(device)
netG.apply(weights_init)
print(netG)

netD = netD().to(device)
netD.apply(weights_init)
print(netD)

print(dataset)

criterion = Hinge()#use Hinge to avoid gredient icrease drasticly
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lrg)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lrd)

lrd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=5, eta_min=5E-6)
lrg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=5, eta_min=5E-6)

RESUME = False
if RESUME:
    path_checkpointD = "./checkpoint/ckpt_latestD_1.pth"  # 断点路径
    path_checkpointG = "./checkpoint/ckpt_latestG_1.pth"  # 断点路径
    checkpointD = torch.load(path_checkpointD)  # 加载断点
    checkpointG = torch.load(path_checkpointG)  # 加载断点
    netD.load_state_dict(checkpointD['net'])  # 加载模型可学习参数

    optimizerD.load_state_dict(checkpointD['optimizer'])  # 加载优化器参数
    start_epoch = checkpointD['epoch']  # 设置开始的epoch



start_epoch = 0 #设置初始epoch大小

for epoch in range(start_epoch + 1, opt.epoch + 1):
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{opt.epoch}', postfix=dict, mininterval=0.3) as pbar:
        for i, (imgs, _) in enumerate(dataloader):
            # 固定生成器G，训练鉴别器D
            imgs = imgs.to(device)
            outputreal = netD(imgs)

            optimizerD.zero_grad()

            noise = torch.randn(opt.batchSize, opt.nz)

            noise = noise.to(device)

            fake = netG(noise)  # 生成假图
            outputfake = netD(fake.detach())  # 避免梯度传到G，因为G不用更新
            lossD = criterion(outputreal, outputfake)
            lossD.backward()
            optimizerD.step()

            # 固定鉴别器D，训练生成器G
            optimizerG.zero_grad()

            output = netD(fake)
            lossG = criterion(output)
            lossG.backward()
            optimizerG.step()

            pbar.set_postfix(**{'lossD': lossD.item(),
                                'lrd': get_lr(optimizerD), 'lossG': lossG.item(),
                                'lrg': get_lr(optimizerG)})
            pbar.update(1)

        #一个epoch
        vutils.save_image(fake.data,
                          '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                          normalize=True)
        lrd_scheduler.step()
        lrg_scheduler.step()

        if epoch % 5 == 0:#每5个epoch，保存一次模型参数.

            checkpointD = {
                "net": netD.state_dict(),
                'optimizer': optimizerD.state_dict(),
                "epoch": epoch
            }
            checkpointG = {
                "net": netG.state_dict(),
                'optimizer': optimizerG.state_dict(),
                "epoch": epoch
            }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpointG, './checkpoint/ckpt_latestG_%s.pth' % (str(epoch)))
            torch.save(checkpointD, './checkpoint/ckpt_latestD_%s.pth' % (str(epoch)))


