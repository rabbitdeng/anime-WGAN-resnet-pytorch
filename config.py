import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--imagesize', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lrd', type=float, default=5e-5,
                    help="Discriminator's learning rate, default=0.00005")  # Discriminator's learning rate
parser.add_argument('--lrg', type=float, default=5e-5,
                    help="Generator's learning rate, default=0.00005")  # Generator's learning rate
parser.add_argument('--data_path', default='data/animedata/', help='folder to train data')  # 将数据集放在此处
parser.add_argument('--outf', default='resnetimg/',
                    help='folder to output images and model checkpoints')  # 输出生成图片以及保存模型的位置


#
opt = parser.parse_args()