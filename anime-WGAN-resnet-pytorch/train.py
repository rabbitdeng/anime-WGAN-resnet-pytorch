import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn

from model import RestNet18, Generator
from losses import Wasserstein


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lrd', type=float, default=5e-5, help='learning rate, default=0.0002')
parser.add_argument('--lrg', type=float, default=5e-5, help='learning rate, default=0.0002')

parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--outf', default='resnetimg/', help='folder to output images and model checkpoints')
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
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)
netD = RestNet18().to(device)
netD.apply(weights_init)
print(netD)
# netG = NetG(opt.ngf)
# netD = NetD(opt.ndf)
# disnet = tm.resnet18(True)

print(dataset)
netG.load_state_dict(torch.load('resnetimg/netG_0025.pth', map_location=device))
netD.load_state_dict(torch.load('resnetimg/netD_0025.pth', map_location=device))
criterionG = Wasserstein()
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lrg)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lrd)
criterionD = Wasserstein()
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
label = label.unsqueeze(1)
# label = label.unsqueeze(1)
# label = label.unsqueeze(1)
start_epoch = 20
for epoch in range(start_epoch + 1, opt.epoch + 1):
    for i, (imgs, _) in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D
        ## 让D尽可能的把真图片判别为1

        imgs = imgs.to(device)
        # imgs = imgs

        outputreal = netD(imgs)

        # label.data.fill_(real_label)
        # label = label.to(device)
        # label = label

        optimizerD.zero_grad()

        ## 让D尽可能把假图片判别为0
        # label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz)
        # noise = torch.randn(opt.batchSize, opt.nz)
        noise = noise.to(device)
        # noise = noise
        fake = netG(noise)  # 生成假图
        outputfake = netD(fake.detach())  # 避免梯度传到G，因为G不用更新

        lossD = criterionD(outputreal, outputfake)
        lossD.backward()
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        # label.data.fill_(real_label)
        # label = label.to(device)
        # label = label
        output = netD(fake)
        lossG = criterionG(output)

        lossG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
          % (epoch, opt.epoch, i, len(dataloader), lossD.item(), lossG.item()))

    vutils.save_image(fake.data,
                  '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                  normalize=True)
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), '%s/netG_%04d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_%04d.pth' % (opt.outf, epoch))
