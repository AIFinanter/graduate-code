import argparse
import os
import numpy as np
import math
import itertools
import sys

from pytorch_ssim import gaussian
import pytorch_ssim

from skimage.metrics import structural_similarity

import torch.nn.functional as F
import copy

QT_DEBUG_PLUGINS = 12

from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from image_pool import *
from utils import *

from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_train = []
loss_train_g = []
loss_train_d = []

x_loss_train = []

loss_test = []

x_loss_test = []
xt = []

loss_test_pixell = []
loss_test_pixelh = []
loss_test_contentl = []
loss_test_contenth = []


class MainThread():
    def __init__(self):
        print("please choose one function to run the task:\n"
              "1.srgan\n"
              "2.esrgan\n"
              "3.esrgan with cycle gan\n")
        choice = input()
        os.makedirs("images/training", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
        self.parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--dataset_name", type=str, default="train", help="name of the dataset")
        self.parser.add_argument("--test_dataset_name", type=str, default="test", help="name of the test dataset")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.9,
                                 help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999,
                                 help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--channels", type=int, default=3, help="R G B")
        self.parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        self.parser.add_argument("--n_cpu", type=int, default=8,
                                 help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
        self.parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
        self.parser.add_argument("--sample_interval", type=int, default=100,
                                 help="interval between saving image samples")
        self.parser.add_argument("--checkpoint_interval", type=int, default=5000,
                                 help="batch interval between model checkpoints")
        self.parser.add_argument("--residual_blocks", type=int, default=23,
                                 help="number of residual blocks in the generator")
        self.parser.add_argument("--warmup_batches", type=int, default=0,
                                 help="number of batches with pixel-wise loss only")
        self.parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
        self.parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
        self.parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
        self.parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
        self.parser.add_argument("--lambda_A", type=float, default=10.0)
        self.parser.add_argument("--lambda_B", type=float, default=10.0)
        self.parser.add_argument("--size", type=int, default=100, help="the size of image pool")
        self.parser.add_argument("--window_size", type=int, default=11, help="the size of the gaussian window")
        self.parser.add_argument("--validation_dataset_name", type=str, default="val",
                                 help="name of the validation dataset")
        self.parser.add_argument("--save_interval", type=int, default=50, help="images to be saved of certain epoch")
        self.parser.add_argument("--test_batch_size", type=int, default=1, help="number of batches used to run test")
        self.opt = self.parser.parse_args()
        print(str(self.opt))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hr_shape = (self.opt.hr_height, self.opt.hr_width)
        lr_shape = (self.opt.hr_height // 4, self.opt.hr_width // 4)

        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_content = torch.nn.L1Loss().to(device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)
        self.criterion_cycle = torch.nn.L1Loss().to(device)
        self.criterion_identity = torch.nn.L1Loss().to(device)
        self.criterion_psnr = torch.nn.MSELoss().to(device)

        choice = int(choice)
        if choice == 1:
            self.srgan()
        elif choice == 2:
            self.esrgan()
        else:
            self.esrganwithcycle()

    def analyze(self, gen_h, imgs_hr, opt, hr_shape, loss_pixel_h, loss_content_h, loss_G_h, loss_G, loss_D, dir):

        loss_psnr = self.criterion_psnr(gen_h, imgs_hr)
        gen_h_numpy = gen_h.cpu().detach().numpy()
        imgs_hr_numpy = imgs_hr.cpu().detach().numpy()

        c1 = 0.01 * 0.01

        ### ssim度量生成图像的质量

        N = opt.batch_size * 3 * opt.hr_height * opt.hr_width
        # 亮度损失
        uxsum = 0
        for i in range(opt.batch_size):
            for j in range(3):
                for x in range(hr_shape[0]):
                    for y in range(hr_shape[1]):
                        uxsum += gen_h_numpy[i][j][x][y]
        ux = uxsum / N

        uysum = 0
        for i in range(opt.batch_size):
            for j in range(3):
                for x in range(hr_shape[0]):
                    for y in range(hr_shape[1]):
                        uysum += imgs_hr_numpy[i][j][x][y]
        uy = uysum / N

        lxy = (2 * ux * uy + c1) / (ux ** 2 + uy ** 2 + c1)

        # 对比度相似性
        sigmaxsum = 0
        for i in range(opt.batch_size):
            for j in range(3):
                for x in range(opt.hr_height):
                    for y in range(opt.hr_width):
                        sigmaxsum += (gen_h_numpy[i][j][x][y] - ux) ** 2
        sigmax2 = sigmaxsum / (N - 1)
        sigmax = np.sqrt(sigmaxsum / (N - 1))

        sigmaysum = 0
        for i in range(opt.batch_size):
            for j in range(3):
                for x in range(opt.hr_height):
                    for y in range(opt.hr_width):
                        sigmaysum += (imgs_hr_numpy[i][j][x][y] - uy) ** 2
        sigmay2 = sigmaysum / (N - 1)
        sigmay = np.sqrt(sigmaysum / (N - 1))
        c2 = 0.03 ** 2

        # 结构相似度

        sxysum = 0

        for i in range(opt.batch_size):
            for j in range(3):
                for x in range(opt.hr_height):
                    for y in range(opt.hr_width):
                        # print("what is imgs_hr_numpy:%d" % imgs_hr_numpy[i][j][x][y])
                        sxysum += (imgs_hr_numpy[i][j][x][y] - ux) * (gen_h_numpy[i][j][x][y] - uy)

        sigmaxy = sxysum / (N - 1)
        sxy = sigmaxy / (sigmax * sigmay)
        ssimxy = ((2 * ux * uy + c1) * (2 * sigmaxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (sigmax2 + sigmay2 + c2))

        ###Mean-SSIM
        # 8*8 gaussian window

        gaussian_weights = gaussian(opt.window_size * opt.window_size, 1.5)

        gw = torch.randn(opt.batch_size, opt.channels, opt.window_size, opt.window_size).to(device)
        #         # print("channels:%d"%opt.channels)

        for x in range(opt.batch_size):
            for y in range(opt.channels):
                for i in range(opt.window_size):
                    for j in range(opt.window_size):
                        gw[x][y][i][j] = gaussian_weights[i * opt.window_size + j]

        #         gw_numpy = gw.cpu().detach().numpy()

        ux_mssim = F.conv2d(gen_h, weight=gw, stride=1, padding=0, groups=1)
        uy_mssim = F.conv2d(imgs_hr, weight=gw, stride=1, padding=0, groups=1)

        ux_mssim2 = ux_mssim.pow(2)
        uy_mssim2 = uy_mssim.pow(2)

        ux2_mssim = F.conv2d(gen_h * gen_h, weight=gw, stride=1, padding=0, groups=1)
        uy2_mssim = F.conv2d(imgs_hr * imgs_hr, weight=gw, stride=1, padding=0, groups=1)
        uxy_mssim = F.conv2d(gen_h * imgs_hr, weight=gw, stride=1, padding=0, groups=1)

        sigmax2_mssim = ux2_mssim - ux_mssim2
        sigmay2_mssim = uy2_mssim - uy_mssim2
        sigmaxy_mssim = uxy_mssim - ux_mssim * uy_mssim

        mssim = (2 * ux_mssim * uy_mssim + c1) * (2 * sigmaxy_mssim + c2) / (
                    (ux_mssim2 + uy_mssim2 + c1) * (sigmax2_mssim + sigmay2_mssim + c2))

        mssim = mssim.sum() / (opt.batch_size * opt.channels * (opt.hr_height - opt.window_size + 1) ** 2)

        if mssim > 1:
            mssim = torch.tensor(1.0)
        elif mssim < -1:
            mssim = torch.tensor(-1.0)
        #         # print("ux2_mssim:")
        #         # print(ux2_mssim)
        #         # print("uy2_mssim")
        #         # print(uy2_mssim)

        #         ux_mssim_numpy = ux_mssim.detach().cpu().numpy()
        #         uy_mssim_numpy = uy_mssim.detach().cpu().numpy()

        # mat = torch.randn(opt.batch_size,opt.channels,opt.window_size,opt.window_size)
        # for x in range(1):
        #     for y in range(opt.channels):
        #         for i in range(opt.window_size):
        #             for j in range(opt.window_size):
        #                 mat[x][y][i][j] = 1

        # print("mat:")
        # print(mat.numpy().shape)
        # mat = torch.Tensor(mat).to(device)
        # mat = torch.randn(1,3,11,11).to(device)
        # gen_h_ex = F.conv2d(gen_h,weight=mat ,stride=1, padding=opt.window_size // 2, groups=1)
        # imgs_hr_ex = F.conv2d(imgs_hr,weight=mat ,stride=1, padding=opt.window_size // 2, groups=1)

        # print("gen_h_ex:")
        # print(gen_h)
        # print(gen_h_ex)
        # print(mat)
        # print(imgs_hr_ex)

        #         gen_h_ex_numpy = gen_h_ex.detach().cpu().numpy()
        #         imgs_hr_ex_numpy = imgs_hr_ex.detach().cpu().numpy()

        # MSE 均方误差
        mse = 0

        # print("batch_size:%d" % opt.batch_size)

        for x in range(opt.batch_size):
            for y in range(opt.channels):
                tmp = 0
                for i in range(opt.hr_height):
                    for j in range(opt.hr_width):
                        tmp += (gen_h_numpy[x][y][i][j] - imgs_hr_numpy[x][y][i][j]) ** 2
                tmp = tmp * (opt.hr_height * opt.hr_width - 1) / (opt.hr_height * opt.hr_width)
                mse += tmp
        mse = mse / (N - opt.batch_size * opt.channels)
        # 将处理图像的loss值传入文件中
        with open('./log/' + dir + '/loss.txt', 'a') as file:
            file.write(str(loss_pixel_h.item()) + ' ')
            file.write(str(loss_content_h.item()) + ' ')
            file.write(str(loss_G_h.item()) + ' ')
            file.write(str(loss_G.item()) + ' ')
            file.write(str(loss_D.item()) + ' ')
            file.write(str(mse) + ' ')
            file.write(str(10 * np.log10(1.0 * 255 * 255 / loss_psnr.item())) + ' ')
            file.write(str(ssimxy.item()) + ' ')
            file.write(str(mssim.item()) + '\n')

    def srgan(self):
        cuda = torch.cuda.is_available()
        hr_shape = (self.opt.hr_height, self.opt.hr_width)
        # test_shape = (128,128)

        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        criterion_content = torch.nn.L1Loss().to(device)
        criterion_pixel = torch.nn.L1Loss().to(device)
        criterion_cycle = torch.nn.L1Loss().to(device)
        criterion_identity = torch.nn.L1Loss().to(device)
        criterion_psnr = torch.nn.MSELoss().to(device)

        # Initialize generator and discriminator
        generator = GeneratorResNet()
        discriminator = SRGANDiscriminator(input_shape=(self.opt.channels, *self.hr_shape))
        feature_extractor = FeatureExtractor()

        # Set feature extractor to inference mode
        feature_extractor.eval()

        # Losses
        criterion_GAN = torch.nn.MSELoss()
        criterion_content = torch.nn.L1Loss()  # 预测值与真实值绝对误差的平均数

        if cuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            feature_extractor = feature_extractor.cuda()
            criterion_GAN = criterion_GAN.cuda()
            criterion_content = criterion_content.cuda()

        if self.opt.epoch != 0:
            # Load pretrained models
            generator.load_state_dict(torch.load("./saved_models/generator_%d.pth"))
            discriminator.load_state_dict(torch.load("./saved_models/discriminator_%d.pth"))

        # Optimizers adam使用SGD原理进行参数优化
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        # 不使用动量算法进行优化
        # optimizer_G = torch.optim.RMSprop(generator.parameters(),lr=opt.lr,beta=(opt.b1,opt.b2))
        # optimizer_D = torch.optim.RMSprop(discriminator.parameters(),lr=opt.lr,beta=(opt.b1,opt.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        dataloader = DataLoader(
            ImageDataset("./%s" % self.opt.dataset_name, hr_shape=self.hr_shape),
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.n_cpu,
        )

        dataloadert = DataLoader(
            ImageDataset("./%s" % self.opt.test_dataset_name, hr_shape=self.hr_shape),
            batch_size=1,
            shuffle=False,
            num_workers=self.opt.n_cpu,
        )

        dataloaderv = DataLoader(
            ImageDataset("./%s" % self.opt.validation_dataset_name, hr_shape=self.hr_shape),
            batch_size=1,
            shuffle=False,
            num_workers=self.opt.n_cpu
        )

        with open('./log/srgan/loss.txt', 'w') as file:
            file.write('PIXEL_LOSS ')
            file.write('CONTENT_LOSS ')
            file.write('ADVERSIAL_LOSS ')
            file.write('GENERATOR_LOSS ')
            file.write('DISCRIMINATOR_LOSS ')
            file.write('MSE ')
            file.write('PSNR ')
            file.write('SSIM ')
            file.write('Mean SSIM')
            file.write('\n')

        # ----------
        #  Training
        # ----------

        cnt = 0

        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            for i, imgs in enumerate(dataloader):
                # Configure model input 把图像数据转换成tensor
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                # Adversarial ground truths
                # requires_grad是参不参与误差反向传播,要不要计算梯度
                valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

                # pixel loss
                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)
                # test_hr = generator(imgs_test)

                loss_pixel_h = self.criterion_pixel(gen_hr, imgs_hr)

                # print(gen_hr.size())
                # Adversarial loss
                loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

                # Content loss
                gen_features = feature_extractor(gen_hr)
                real_features = feature_extractor(imgs_hr)
                loss_content = criterion_content(gen_features, real_features.detach())

                # # WGAN改进:每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
                # for parm in discriminator.parameters():
                #     parm.data.clamp_(-opt.clamp_num, opt.clamp_num)

                # Total loss
                loss_G = loss_content + 1e-3 * loss_GAN

                loss_G.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # srgan的loss函数不含对数,不需要改进
                # Loss of real and fake images valid设置为1也就是高分辨率图像认为是1 fake设置为0也就是低分辨率图像设置为0
                loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

                # Total loss:这就是判别器损失
                loss_D = (loss_real + loss_fake) / 2

                # 反向传播
                loss_D.backward()
                optimizer_D.step()

                self.analyze(gen_hr, imgs_hr, self.opt, self.hr_shape, loss_pixel_h, loss_content, loss_GAN, loss_G,
                             loss_D, 'srgan')

                # --------------
                #  Log Progress
                # --------------

                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
                )

                batches_done = epoch * len(dataloader) + i

                if batches_done % self.opt.sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

                    # print(" last2 gen_hr")
                    # print(gen_hr.size())
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)

                    # print("final--------\n")
                    # print("imgs_lr")
                    # print(imgs_lr.size())
                    # print("imgs_hr")
                    # print(imgs_hr.size())
                    #
                    # print("gen_hr")
                    # print(gen_hr.size())
                    # 值得注意的是左边一列是imgs_lr 右边一列是gen_hr 再右边一列我们把原高分辨图输出

                    img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
                    save_image(img_grid, "./images/srgan/train/%d.png" % batches_done, normalize=False)

            for a, imgsv in enumerate(dataloaderv):
                imgsv_l = Variable(imgsv["lr"].type(Tensor))
                imgsv_h = Variable(imgsv["hr"].type(Tensor))
                genv_h = generator(imgsv_l)
                genv_h = make_grid(genv_h, nrow=1, noemalize=True)
                save_image(genv_h, "./validation/srgan/pic/%s" % str(epoch))
                ssim = pytorch_ssim.ssim(imgsv_h, genv_h)
                with open('./validation/srgan/text/%s' % str(epoch), 'a') as file:
                    file.write(ssim + '\n')

            if self.opt.checkpoint_interval != -1 and epoch % self.opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "./saved_models/srgan_generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), "./saved_models/srgan_discriminator_%d.pth" % epoch)

    def esrgan(self):
        os.makedirs("images/training", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)

        opt = self.opt
        print(opt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hr_shape = (opt.hr_height, opt.hr_width)

        # Initialize generator and discriminator
        g = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
        discriminator = ESRGANDiscriminator(input_shape=(opt.channels, *hr_shape)).to(device)
        feature_extractor = FeatureExtractor().to(device)

        # Set feature extractor to inference mode
        feature_extractor.eval()

        # Losses
        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        criterion_content = torch.nn.L1Loss().to(device)
        criterion_pixel = torch.nn.L1Loss().to(device)

        if opt.epoch != 0:
            # Load pretrained models
            g.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
            discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

        # Optimizers
        optimizer_G = torch.optim.Adam(g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        dataloader = DataLoader(
            ImageDataset("./%s" % opt.dataset_name, hr_shape=hr_shape),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )

        dataloadert = DataLoader(
            ImageDataset("./%s" % opt.test_dataset_name, hr_shape=self.hr_shape),
            batch_size=opt.test_batch_size,
            shuffle=True,
            num_workers=opt.n_cpu
        )

        dataloaderv = DataLoader(
            ImageDataset("./%s" % opt.validation_dataset_name, hr_shape=self.hr_shape),
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.n_cpu
        )

        # ----------
        #  Training
        # ----------

        plt.figure(num="trainning loss of esrgan", figsize=(1080, 1080), facecolor='red', edgecolor='black')
        plt.rcParams['font.family'] = 'MicroSoft YaHei'  # 设置字体，默认字体显示不了中文

        with open('./log/esrgan/train/loss.txt', 'w') as file:
            file.write('PIXEL_LOSS ')
            file.write('CONTENT_LOSS ')
            file.write('ADVERSIAL_LOSS ')
            file.write('GENERATOR_LOSS ')
            file.write('DISCRIMINATOR_LOSS ')
            file.write('MSE ')
            file.write('PSNR ')
            file.write('SSIM ')
            file.write('Mean SSIM')

        for epoch in range(opt.epoch, opt.n_epochs):
            for i, imgs in enumerate(dataloader):

                batches_done = epoch * len(dataloader) + i
                if batches_done >= 500:
                    x_loss_train.append(batches_done)

                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_h = g(imgs_lr)

                # save_image(imgs_lr,"./esrgan_lr_groundtruth/%d"%batches_done,normalize=False)
                # save_image(imgs_hr,"./esrgan_hr_groundtruth/%d"%batches_done,normalize=False)
                gen_hi = make_grid(gen_h, nrow=1, normalize=True)
                save_image(gen_hi, "esrgan_hr_generated/%d.png" % batches_done, normalize=True)

                # Measure pixel-wise loss against ground truth
                loss_pixel = criterion_pixel(gen_h, imgs_hr)

                if batches_done < opt.warmup_batches:
                    # Warm-up (pixel-wise loss only)
                    loss_pixel.backward()
                    optimizer_G.step()
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
                    )
                    continue

                # Extract validity predictions from discriminator
                pred_real = discriminator(imgs_hr).detach()
                pred_fake = discriminator(gen_h)

                # Adversarial loss (relativistic average GAN)
                loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

                # Content loss
                gen_features = feature_extractor(gen_h)
                real_features = feature_extractor(imgs_hr).detach()
                loss_content = criterion_content(gen_features, real_features)

                # Total generator loss
                loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

                loss_G.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                pred_real = discriminator(imgs_hr)
                pred_fake = discriminator(gen_h.detach())

                # Adversarial loss for real and fake images (relativistic average GAN)
                loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2
                loss_train.append(loss_D)

                loss_D.backward()
                optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_content.item(),
                        loss_GAN.item(),
                        loss_pixel.item(),
                    )
                )

                self.analyze(gen_h, imgs_hr, opt, hr_shape, loss_pixel, loss_content, loss_GAN, loss_G, loss_D,
                             'esrgan')

                if batches_done > 1000:
                    for i, imgst in enumerate(dataloadert):
                        imgst_lr = Variable(imgst["lr"].type(Tensor)).to(device)
                        imgst_hr = Variable(imgst["hr"].type(Tensor)).to(device)

                        gen_ht = g(imgst_lr)
                        # fen_lt = f(imgst_hr)
                        # valid_lt = Variable(Tensor(np.ones((imgs_lr.size(0), *disl.output_shape))),
                        #                     requires_grad=False)
                        # fake_lt = Variable(Tensor(np.zeros((imgs_lr.size(0), *disl.output_shape))),
                        #                    requires_grad=False)
                        valid_ht = Variable(Tensor(np.ones((imgs_hr.size(0), *discriminator.output_shape))),
                                            requires_grad=False)
                        fake_ht = Variable(Tensor(np.zeros((imgs_hr.size(0), *discriminator.output_shape))),
                                           requires_grad=False)
                        loss_pixel_ht = criterion_pixel(gen_h, imgs_hr)
                        # loss_pixel_lt = criterion_pixel(fen_l, imgs_lr)

                        # Content loss
                        gen_features_ht = feature_extractor(gen_ht)
                        real_features_ht = feature_extractor(imgst_hr).detach()
                        loss_content_ht = criterion_content(gen_features, real_features)

                        # gen_features_lt = feature_extractor(fen_lt)
                        # real_features_lt = feature_extractor(imgst_lr).detach()
                        # loss_content_lt = criterion_content(gen_features_lt, real_features_lt)

                        if i < 10 or i % 100 == 0:
                            imgst_lr = nn.functional.interpolate(imgst_lr, scale_factor=4)
                            gen_lt = nn.finctional.interpolate(gen_lt, scale_factor=4)
                            img_grid = denormalize(torch.cat((imgs_lr, imgs_hr, gen_h), -1))
                            # print("push through hell")
                            save_image(img_grid, "./images/esrgan/test/%dv%d.png" % (batches_done, i), nrow=1,
                                       normalize=False)

                        if i == 0:
                            xt.append(batches_done)
                            loss_test_pixelh.append(loss_pixel_ht)
                            # loss_test_pixelh.append(loss_pixel_lt)
                        self.printf(
                            "Test Set Performance [Epoch %d/%d] [Batch %d/%d] [pixel-loss: %f] [content loss: %f]"
                            % (
                                epoch,
                                self.opt.n_epochs,
                                i,
                                len(dataloadert),
                                loss_pixel_ht.item(),
                                # loss_pixel_lt.item(),
                                loss_content_ht.item(),
                                # loss_content_lt.item()
                            )
                        )

                        if batches_done >= 138000:
                            writer = SummaryWriter('./board/esrgan/test', comment='selection of a bunch of pictures')
                            for i in xt:
                                writer.add_scalars('test loss',
                                                   {'pixel-wise loss': loss_pixel_ht[i],
                                                    # 'low-res pixel-wise loss': pixel_lt[i],
                                                    'content loss': loss_content_ht[i],
                                                    # 'discriminator loss': d_h[i],
                                                    # 'low-res discriminator loss': d_l[i]
                                                    })
                                writer.add_histogram('pixel', loss_pixel_ht[i], i)
                                writer.add_histogram('content', loss_content_ht[i], i)
                                # writer.add_histogram('d_h', d_h[i], i)
                                # writer.add_histogram('d_l', d_l[i], i)

                            writer.close()

                if batches_done % opt.sample_interval == 0:
                    # Save image grid with upsampled inputs and ESRGAN outputs
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                    img_grid = denormalize(torch.cat((imgs_lr, imgs_hr, gen_h), -1))
                    save_image(img_grid, "images/esrgan/train/%d.png" % batches_done, nrow=1, normalize=False)

                if batches_done % opt.checkpoint_interval == 0:
                    # Save model checkpoints
                    torch.save(g.state_dict(), "saved_models/generator_%d.pth" % epoch)
                    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)

            #                 if batches_done % 1000 == 0:
            #                     # save training loss
            #                     plt.plot(x_loss_train, loss_train_g, '.', label="generator train loss")
            #                     plt.plot(x_loss_train, loss_train_d, '.', label="discriminator train loss")

            #                     plt.savefig('./loss_pictires/loss_train' + "%" % batches_done // 1000)

            for a, imgsv in enumerate(dataloaderv):
                imgsv_l = Variable(imgsv["lr"].type(Tensor))
                imgsv_h = Variable(imgsv["hr"].type(Tensor))
                genv_h = g(imgsv_l)
                genv_h = make_grid(genv_h, nrow=1, noemalize=True)
                save_image(genv_h, "./validation/esrgan/pic/%s.png" % str(epoch))
                ssimtmp = pytorch_ssim.ssim(imgsv_h, genv_h)
                with open('./validation/esrgan/text/%s' % str(epoch)) as file:
                    file.write(ssimtmp + '\n')

    def esrganwithcycle(self):
        g = GeneratorRRDB(3).to(device)
        f = Generator(mode='hl').to(device)

        dish = Discriminator(mode='h').to(device)
        disl = Discriminator(mode='l').to(device)

        # Losses : grades matters
        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        criterion_content = torch.nn.L1Loss().to(device)
        criterion_pixel = torch.nn.L1Loss().to(device)
        criterion_cycle = torch.nn.L1Loss().to(device)
        criterion_identity = torch.nn.L1Loss().to(device)
        criterion_psnr = torch.nn.MSELoss().to(device)
        feature_extractor = FeatureExtractor().to(device)
        # Set feature extractor to inference mode
        feature_extractor.eval()

        if self.opt.epoch != 0:
            # Load pretrained models
            g.load_state_dict(torch.load("saved_models/generator_LH%d.pth" % self.opt.epoch))
            f.load_state_dict(torch.load("saved_models/generator_HL%d.pth" % self.opt.epoch))
            disl.load_state_dict(torch.load("saved_models/discriminator_L%d.pth" % self.opt.epoch))
            dish.load_state_dict(torch.load("saved_models/discriminator_H%d.pth" % self.opt.epoch))

        # Optimizers
        """
        注意:itertools是Python中的一个模块，具有用于处理迭代器的函数集合。
        它们非常容易地遍历列表和字符串之类的可迭代对象。 
        chain()是这样的itertools函数之一。
        """

        optimizer_generator = torch.optim.Adam(
            itertools.chain(g.parameters(), f.parameters()), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2)
        )
        optimizer_dl = torch.optim.Adam(disl.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_dh = torch.optim.Adam(dish.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        # learning rate update schedulers
        lr_scheduler_generator = torch.optim.lr_scheduler.LambdaLR(
            optimizer_generator, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )

        lr_scheduler_dl = torch.optim.lr_scheduler.LambdaLR(
            optimizer_dl, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )

        lr_scheduler_dh = torch.optim.lr_scheduler.LambdaLR(
            optimizer_dh, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )

        # Data loader : train dataloader and test dataloader (我的修改是删掉transforms)
        dataloader = DataLoader(
            ImageDataset("./%s" % self.opt.dataset_name, self.hr_shape, unaligned=True),
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=5,
        )

        # test dataloader may be modified after a while
        dataloadert = DataLoader(
            ImageDataset("./%s" % self.opt.test_dataset_name, self.hr_shape, unaligned=True, mode="test"),
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=1,
        )

        # validation dataloader mainly to generate gif image
        dataloaderv = DataLoader(
            ImageDataset("./%s" % self.opt.validation_dataset_name, self.hr_shape, unaligned=True, mode="test"),
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=1,
        )

        def sample_images(batches_done):
            """Saves a generated sample from the test set"""
            imgs = next(iter(dataloadert))
            g.eval()
            f.eval()
            real_L = Variable(imgs["lr"].type(Tensor))
            fake_H = g(real_L)
            real_H = Variable(imgs["hr"].type(Tensor))
            fake_L = f(imgs["hr"].type(Tensor))
            # arange images along y-axis
            image_grid = torch.cat((real_L, fake_H, real_H, fake_L), 1)
            save_image(image_grid, "cy_images/%s.png" % (self.opt.dataset_name), normalize=False)

        def backward_D_loss(netD, real, fake):
            """
            Calculate GAN loss for the discriminator
                        Parameters:
                        netD (network)      -- the discriminator D
                        real (tensor array) -- real images
                        fake (tensor array) -- images generated by a generator
                    Return the discriminator loss.
                    We also call loss_D.backward() to calculate the gradients.
            """
            #         valid_l = Variable(Tensor(np.ones((imgs_lr.size(0), *disl.output_shape))), requires_grad=False)
            #         fake_l = Variable(Tensor(np.zeros((imgs_lr.size(0), *disl.output_shape))), requires_grad=False)
            #         valid_h = Variable(Tensor(np.ones((imgs_hr.size(0), *dish.output_shape))), requires_grad=False)
            #         fake_h = Variable(Tensor(np.zeros((imgs_hr.size(0), *dish.output_shape))), requires_grad=False)

            # Real
            pred_real = netD(real)
            loss_D_real = criterion_GAN(pred_real,
                                        Variable(Tensor(np.ones((real.size(0), *netD.output_shape))).to(device),
                                                 requires_grad=False))  # True
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = criterion_GAN(pred_fake,
                                        Variable(Tensor(np.zeros((real.size(0), *netD.output_shape))).to(device),
                                                 requires_grad=False))
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            loss_train_d.append(loss_D)
            return loss_D

        lr_transform = transforms.Compose(
            [
                transforms.Resize((self.opt.hr_height // 4, self.opt.hr_height // 4), Image.BICUBIC),
                # transforms.Resize(opt.hr_height//4,opt.hr_width//4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        hr_transform = transforms.Compose(
            [
                transforms.Resize((self.opt.hr_height, self.opt.hr_width), Image.BICUBIC),
                # transform.Resize(opt.hr_height,opt.hr_width),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        # ----------
        #  Training
        # ----------

        plt.figure(num="trainning loss of esrgan", figsize=(10, 10), facecolor='red', edgecolor='black')
        plt.rcParams['font.family'] = 'MicroSoft YaHei'  # 设置字体，默认字体显示不了中文

        # print("start training...")

        # for epoch in range(opt.epoch, opt.n_epochs):
        #     for i, imgs in enumerate(dataloader):
        #
        #         batches_done = epoch * len(dataloader) + i
        #         if batches_done >= 500:
        #             x_loss_train.append(batches_done)
        #
        #         # Configure model input
        #         imgs_lr = Variable(imgs["lr"].type(Tensor))
        #         imgs_hr = Variable(imgs["hr"].type(Tensor))
        #
        #         # Adversarial ground truths
        #         valid_l = Variable(Tensor(np.ones((imgs_lr.size(0), *disl.output_shape))), requires_grad=False)
        #         fake_l = Variable(Tensor(np.zeros((imgs_lr.size(0), *disl.output_shape))), requires_grad=False)
        #         valid_h = Variable(Tensor(np.ones((imgs_hr.size(0), *dish.output_shape))), requires_grad=False)
        #         fake_h = Variable(Tensor(np.zeros((imgs_hr.size(0), *dish.output_shape))), requires_grad=False)
        #
        #         # ------------------
        #         #  Train Generators
        #         # ------------------
        #         g.train()
        #         f.train()
        #
        #         print("F and G already trained")
        #
        #         optimizer_generator.zero_grad()
        #
        #         # Generate a high resolution image from low resolution input
        #         gen_h = g(imgs_lr)
        #         gen_l = f(imgs_hr)
        #
        #         # identity loss
        #         loss_id_h = criterion_identity(gen_h, imgs_hr)
        #         loss_id_l = criterion_identity(gen_l, imgs_lr)
        #
        #         loss_identity = (loss_id_h + loss_id_l) / 2
        #
        #         # Measure pixel-wise loss against ground truth
        #         loss_pixel_h = criterion_pixel(gen_h, imgs_hr)
        #         loss_pixel_l = criterion_pixel(gen_l, imgs_lr)
        #
        #         print("warming,please wait")
        #
        #         if batches_done < opt.warmup_batches:
        #             # Warm-up (pixel-wise loss only)
        #             loss_pixel_l.backward()
        #             loss_pixel_h.backward()
        #             optimizer_generator.step()
        #             print(
        #                 "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
        #                 % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel_h.item())
        #             )
        #             continue
        #
        #         print("beyond challenge")
        #         # Extract validity predictions from discriminator
        #         print("接下来输出:size of imgs_hr:")
        #         print(imgs_hr.size())
        #         print("接下来输出:size of gen_h:")
        #         print(gen_h.size())
        #         pred_real = dish(imgs_hr).detach()
        #         pred_fake = dish(gen_h)
        #
        #         # Adversarial loss (relativistic average GAN)
        #         loss_GAN_h = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid_h)
        #         loss_GAN_l = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid_l)
        #
        #         # Content loss
        #         gen_features_h = feature_extractor(gen_h)
        #         real_features_h = feature_extractor(imgs_hr).detach()
        #         loss_content_h = criterion_content(gen_features_h, real_features_h)
        #
        #         gen_features_l = feature_extractor(gen_l)
        #         real_features_l = feature_extractor(imgs_lr).detach()
        #         loss_content_l = criterion_content(gen_features_l, real_features_l)
        #
        #         # Total generator loss
        #         loss_G = loss_content_h + opt.lambda_adv * loss_GAN_h + opt.lambda_pixel * loss_pixel_h
        #
        #         loss_G.backward()
        #         optimizer_generator.step()
        #
        #         # ---------------------
        #         #  Train Discriminator
        #         # ---------------------
        #
        #         optimizer_dl.zero_grad()
        #
        #         pred_real_h = dish(imgs_hr)
        #         pred_fake_h = dish(gen_h.detach())
        #         pred_real_l = disl(imgs_lr)
        #         pred_fake_l = disl(gen_l.detach())
        #
        #         # Adversarial loss for real and fake images (relativistic average GAN)
        #         loss_real = criterion_GAN(pred_real_h - pred_fake_h.mean(0, keepdim=True), valid_h)
        #         loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake_h)
        #
        #         # Total loss
        #         loss_D = (loss_real + loss_fake) / 2
        #         loss_train.append(loss_D)
        #
        #         loss_D.backward()
        #         optimizer_dl.step()
        #         optimizer_dh.step()
        #         # --------------
        #         #  Log Progress
        #         # --------------
        #
        #         print(
        #             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
        #             % (
        #                 epoch,
        #                 opt.n_epochs,
        #                 i,
        #                 len(dataloader),
        #                 loss_D.item(),
        #                 loss_G.item(),
        #                 loss_content_h.item(),
        #                 loss_GAN_h.item(),
        #                 loss_pixel_h.item(),
        #             )
        #         )
        #
        #         if batches_done % opt.sample_interval == 0:
        #             # Save image grid with upsampled inputs and ESRGAN outputs
        #             imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        #             img_grid = denormalize(torch.cat((imgs_lr, gen_h), -1))
        #             print("push through hell")
        #             save_image(img_grid, "./training/%d.png" % batches_done, nrow=1, normalize=False)
        #
        #         if batches_done % opt.checkpoint_interval == 0:
        #             # Save model checkpoints
        #             torch.save(g.state_dict(), "saved_models/generator_LH%d.pth" % epoch)
        #             torch.save(f.state_dict(), "saved_models/fgenerator_HL%d.pth" % epoch)
        #             torch.save(disl.state_dict(), "saved_models/discriminator_L%d.pth" % epoch)
        #             torch.save(dish.state_dict(), "saved_models/discriminator_H%d.pth" % epoch)
        #
        #         if batches_done % 1000 == 0:
        #             # save training loss
        #             plt.plot(x_loss_train, loss_train_g, '.', label="generator train loss")
        #             plt.plot(x_loss_train, loss_train_d, '.', label="discriminator train loss")
        #             plt.savefig('./loss_pictires/loss_train' + "%s" % batches_done // 1000)

        imagepool = ImagePool(100)

        """
            file.write(loss_pixel_h.item() + ' ')
            file.write(loss_content_h.item() + ' ')
            file.write(loss_G_h.item()+' ')
            file.write(loss_G.item()+' ')
            file.write(loss_D.item()+' ')
            file.write(10 * np.log10(1.0 * 255 * 255 / loss_psnr) + ' ')
            file.write(ssimxy+' ')
            file.write(mssim)
            file.write(mse)
        """
        with open('./log/esrganwithcycle/train/loss.txt', 'w') as file:
            file.write('PIXEL_LOSS ')
            file.write('CONTENT_LOSS ')
            file.write('ADVERSIAL_LOSS ')
            file.write('GENERATOR_LOSS ')
            file.write('DISCRIMINATOR_LOSS ')
            file.write('MSE ')
            file.write('PSNR ')
            file.write('SSIM ')
            file.write('Mean SSIM\n')

        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            for i, imgs in enumerate(dataloader):
                batches_done = epoch * len(dataloader) + i
                if batches_done >= 500:
                    x_loss_train.append(batches_done)

                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(Tensor)).to(device)
                imgs_hr = Variable(imgs["hr"].type(Tensor)).to(device)

                # Adversarial ground truths
                valid_l = Variable(Tensor(np.ones((imgs_lr.size(0), *disl.output_shape))), requires_grad=False).to(
                    device)
                fake_l = Variable(Tensor(np.zeros((imgs_lr.size(0), *disl.output_shape))), requires_grad=False).to(
                    device)
                valid_h = Variable(Tensor(np.ones((imgs_hr.size(0), *dish.output_shape))), requires_grad=False).to(
                    device)
                fake_h = Variable(Tensor(np.zeros((imgs_hr.size(0), *dish.output_shape))), requires_grad=False).to(
                    device)

                # ------------------
                #  Train Generators
                # ------------------
                g.train()
                f.train()

                # print("F and G already trained")

                optimizer_generator.zero_grad()

                # Generate a high resolution image from low resolution input (G(A) G(B))
                # print("imgs_lr:")
                # print(imgs_lr.size())
                # print("imgs_hr")
                # print(imgs_hr.size())

                gen_h = g(imgs_lr).to(device)
                fen_l = f(imgs_hr).to(device)
                # print("gen_h")
                # print(gen_h.size())
                # print("fen_l")
                # print(fen_l.size())

                # save_image(imgs_lr,"./esrganwithcycle_lr_groundtruth/%d.png"%batches_done,normalize=False)
                # save_image(imgs_hr,"./esrganwithcycle_hr_groundtruth/%d.png"%batches_done,normalize=False)
                save_image(gen_h, "./ersganwithcycle_hr_generated/%d.png" % batches_done, normalize=False)
                save_image(fen_l, "./esrganwithcycle_lr_generated/%d.png" % batches_done, normalize=False)

                # D(A) D(B)
                dis_l = disl(imgs_lr).to(device)
                dis_h = dish(imgs_hr).to(device)

                # D(G(A)) D(F(B))
                dis_gh = dish(gen_h).to(device)
                dis_fl = disl(fen_l).to(device)

                # F(G(A)) G(F(B))
                fen_gh = f(gen_h).to(device)
                gen_fl = g(fen_l).to(device)

                # # F(A) G(B) 减少颜色纹理损失(我直接把这个函数优化掉)
                # fen_h = f(hr_transform(imgs_lr))
                # gen_l = g(lr_transform(imgs_hr))

                pool_l = ImagePool(self.opt.size)
                pool_h = ImagePool(self.opt.size)

                pool_l.query(fen_l)
                pool_h.query(gen_h)

                optimizer_generator.zero_grad()
                """Calculate the loss for generators G_A and G_B"""
                lambda_idt = self.opt.lambda_id
                lambda_A = self.opt.lambda_A
                lambda_B = self.opt.lambda_B

                # 我的训练集完全对应，不考虑identity loss
                # # Identity loss
                # if lambda_idt > 0:
                #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
                #     idt_hr = g(lr_transform(imgs_hr))
                #     loss_idt_hr = criterion_identity(idt_hr, imgs_hr) * lambda_B * lambda_idt
                #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
                #     idt_lr = f(hr_transform(imgs_lr))
                #     loss_idt_lr = criterion_identity(idt_lr, imgs_lr) * lambda_A * lambda_idt
                # else:
                #     loss_idt_hr = 0
                #     loss_idt_lr = 0

                # pixel loss
                loss_pixel_h = criterion_pixel(gen_h, imgs_hr)
                loss_pixel_l = criterion_pixel(fen_l, imgs_lr)

                # Content loss
                gen_features_h = feature_extractor(gen_h)
                real_features_h = feature_extractor(imgs_hr).detach()
                loss_content_h = criterion_content(gen_features_h, real_features_h)

                gen_features_l = feature_extractor(fen_l)
                real_features_l = feature_extractor(imgs_lr).detach()
                loss_content_l = criterion_content(gen_features_l, real_features_l)

                # Adversarial loss(relativistic average GAN)
                # GAN loss D_A(G_A(A))
                loss_G_h = criterion_GAN(dis_gh - dish(imgs_hr).mean(0, keepdim=True), valid_h)
                # GAN loss D_B(G_B(B))
                loss_G_l = criterion_GAN(dis_fl - disl(imgs_lr).mean(0, keepdim=True), valid_l)

                # Forward cycle loss || G_B(G_A(A)) - A||
                loss_cycle_l = criterion_cycle(fen_gh, imgs_lr) * 0.5
                # Backward cycle loss || G_A(G_B(B)) - B||
                loss_cycle_h = criterion_cycle(gen_fl, imgs_hr) * 0.5
                # combined loss and calculate gradients
                loss_G = 1.8 * loss_G_h + 0.2 * loss_G_l + loss_cycle_l + loss_cycle_h + 1.9 * loss_content_h + loss_content_l + loss_pixel_l * 0.5 + loss_pixel_h * 0.5
                loss_G.backward()

                loss_train_g.append(loss_G)

                optimizer_generator.step()

                optimizer_dl.zero_grad()
                optimizer_dh.zero_grad()
                """Calculate the loss of Discriminator A and Discriminator B"""
                """Calculate GAN loss for Discriminator A"""

                loss_dis_h = backward_D_loss(dish, imgs_hr, gen_h)
                loss_dis_l = backward_D_loss(disl, imgs_lr, fen_l)

                optimizer_dl.step()
                optimizer_dh.step()

                # --------------
                #  Log Progress
                # --------------

                self.analyze(gen_h, imgs_hr, self.opt, self.opt.hr_shape, loss_pixel_h, loss_G_h, loss_G, loss_dis_h,
                             'esrganwithcycle')

                self.printf(
                    "[Epoch %d/%d] [Batch %d/%d] [high-res D loss: %f] [low-res D loss: %f] [G loss: %f,low2high adv: %f,high2low adv: %f] [high-res pixel-loss: %f] [low-res pixel-loss: %f] [high-res content-loss: %f] [low-res content loss: %f]"
                    % (
                        epoch,
                        self.opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_dis_h.item(),
                        loss_dis_l.item(),
                        loss_G.item(),
                        loss_G_h.item(),
                        loss_G_l.item(),
                        loss_pixel_h.item(),
                        loss_pixel_l.item(),
                        loss_content_h.item(),
                        loss_content_l.item()
                    )
                )
                if batches_done > 1000:
                    for i, imgst in enumerate(dataloadert):
                        imgst_lr = Variable(imgst["lr"].type(Tensor)).to(device)
                        imgst_hr = Variable(imgst["hr"].type(Tensor)).to(device)

                        gen_ht = g(imgst_lr)
                        fen_lt = f(imgst_hr)
                        valid_lt = Variable(Tensor(np.ones((imgs_lr.size(0), *disl.output_shape))),
                                            requires_grad=False)
                        fake_lt = Variable(Tensor(np.zeros((imgs_lr.size(0), *disl.output_shape))),
                                           requires_grad=False)
                        valid_ht = Variable(Tensor(np.ones((imgs_hr.size(0), *dish.output_shape))),
                                            requires_grad=False)
                        fake_ht = Variable(Tensor(np.zeros((imgs_hr.size(0), *dish.output_shape))),
                                           requires_grad=False)
                        loss_pixel_ht = criterion_pixel(gen_h, imgs_hr)
                        loss_pixel_lt = criterion_pixel(fen_l, imgs_lr)

                        # Content loss
                        gen_features_ht = feature_extractor(gen_ht)
                        real_features_ht = feature_extractor(imgst_hr).detach()
                        loss_content_ht = criterion_content(gen_features_h, real_features_h)

                        gen_features_lt = feature_extractor(fen_lt)
                        real_features_lt = feature_extractor(imgst_lr).detach()
                        loss_content_lt = criterion_content(gen_features_lt, real_features_lt)

                        if i < 10 or i % 100 == 0:
                            imgst_lr = nn.functional.interpolate(imgst_lr, scale_factor=4)
                            gen_lt = nn.finctional.interpolate(gen_lt, scale_factor=4)
                            img_grid = denormalize(torch.cat((imgs_lr, imgs_hr, fen_l, gen_h), -1))
                            # print("push through hell")
                            save_image(img_grid, "./testing_cyc/%dv%d.png" % (batches_done, i), nrow=1,
                                       normalize=False)

                        if i == 0:
                            xt.append(batches_done)
                            loss_test_pixelh.append(loss_pixel_ht)
                            loss_test_pixell.append(loss_pixel_lt)
                        self.printf(
                            "Test Set Performance [Epoch %d/%d] [Batch %d/%d] [high-res pixel-loss: %f] [low-res pixel-loss: %f] [high-res content-loss: %f] [low-res content loss: %f]"
                            % (
                                epoch,
                                self.opt.n_epochs,
                                i,
                                len(dataloadert),
                                loss_pixel_ht.item(),
                                loss_pixel_lt.item(),
                                loss_content_ht.item(),
                                loss_content_lt.item()
                            )
                        )

                        if batches_done >= 138000:
                            writer = SummaryWriter('./board/esrganwithcycle/test',
                                                   comment='selection of a bunch of pictures')
                            for i in xt:
                                writer.add_scalars('test loss', {'high-res pixel-wise loss': loss_pixel_ht[i],
                                                                 'low-res pixel-wise loss': loss_pixel_lt[i],
                                                                 # 'high-res discriminator loss': d_h[i],
                                                                 # 'low-res discriminator loss': d_l[i]
                                                                 'high-res content loss': loss_content_ht[i],
                                                                 'low-res content loss': loss_content_lt[i]
                                                                 })
                                writer.add_histogram('high-res pixel', loss_pixel_ht[i], i)
                                writer.add_histogram('low-res pixel', loss_pixel_lt[i], i)
                                writer.add_histogram('high-content loss', loss_content_ht[i])
                                writer.add_histogram('low-content loss', loss_content_lt[i])
                                # writer.add_histogram('d_h', d_h[i], i)
                                # writer.add_histogram('d_l', d_l[i], i)

                            writer.close()

                if batches_done % self.opt.sample_interval == 0:
                    # Save image grid with upsampled inputs and ESRGAN outputs
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                    img_grid = denormalize(torch.cat((imgs_lr, gen_h), -1))
                    # print("push through hell")
                    save_image(img_grid, "./1/%d.png" % batches_done, nrow=1, normalize=False)

                if batches_done % self.opt.checkpoint_interval == 0:
                    # Save model checkpoints
                    torch.save(g.state_dict(), "saved_models/generator_LH%d.pth" % epoch)
                    torch.save(f.state_dict(), "saved_models/fgenerator_HL%d.pth" % epoch)
                    torch.save(disl.state_dict(), "saved_models/discriminator_L%d.pth" % epoch)
                    torch.save(dish.state_dict(), "saved_models/discriminator_H%d.pth" % epoch)

                if batches_done % 1000 == 0:
                    # save training loss
                    plt.plot(x_loss_train, loss_train_g, '.', label="generator train loss")
                    s = len(loss_train_d)
                    for i in range(0, s, 2):
                        loss_train_d[i / 2] = loss_train_d[i] * 0.5 + loss_train_d[i + 1] * 0.5

                    plt.plot(x_loss_train, loss_train_d, '.', label="discriminator train loss")

                    plt.savefig('./loss_pictures/loss_train' + "%s" % str(batches_done // 1000))

            for a, imgsv in enumerate(dataloaderv):
                imgsv_l = Variable(imgsv["lr"].type(Tensor))
                imgsv_h = Variable(imgsv["hr"].type(Tensor))
                genv_h = g(imgsv_l)
                genv_h = make_grid(genv_h, nrow=1, noemalize=True)
                save_image(genv_h, "./validation/esrganwithcycle/pic/%s" % str(epoch))
                ssimtmp = pytorch_ssim.ssim(imgsv_h, genv_h)
                with open('./validation/esrganwithcycle/text/%s' % str(epoch)) as file:
                    file.write(ssimtmp + '\n')
                    # file.write(str(mssim.item())+'\n')
        # ux uy sigma_x sigma_y
        # ux_mssim = []
        # uy_mssim = []
        # sigmax2_mssim = []
        # sigmay2_mssim = []
        # sigmaxy_mssim = []
        # matssim = []
        # for i in range(opt.batch_size):
        #     tmp = []
        #     for j in range(opt.channels):
        #         tmp.append([])
        #     ux_mssim.append(copy.deepcopy(tmp))
        #     uy_mssim.append(copy.deepcopy(tmp))
        #     sigmax2_mssim.append(copy.deepcopy(tmp))
        #     sigmay2_mssim.append(copy.deepcopy(tmp))
        #     sigmaxy_mssim.append(copy.deepcopy(tmp))
        #     matssim.append(copy.deepcopy(tmp))


"""    
#         ux_mssim = [[] for range(opt.channels)] for j in range(opt.batch_size)
#         uy_mssim = [[] for range(opt.channels)] for j in range(opt.batch_size)
#         sigmax2_mssim = [[] for range(opt.channels)] for j in range(opt.batch_size)
#         sigmay2_mssim = [[] for range(opt.channels)] for j in range(opt.batch_size)
#         sigmaxy_mssim = [[] for range(opt.channels)] for j in range(opt.batch_size)
"""
#         for b in range(opt.batch_size):
#             for c in range(opt.channels):
#                 for posx in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                     for posy in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                         tmp = 0
#                         for i in range(posx-opt.window_size//2,posx+opt.window_size//2+1):
#                             for j in range(posy-opt.window_size//2,posy+opt.window_size//2+1):
#                                 xi = i - posx + opt.window_size//2
#                                 yi = j - posy + opt.window_size//2
#                                 tmp += gaussian_weights[xi*opt.window_size+yi]*gen_h[b][c][i][j]
#                         ux_mssim[b][c].append(tmp.item())

#         for b in range(opt.batch_size):
#             for c in range(opt.channels):
#                 for posx in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                     for posy in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                         tmp = 0
#                         for i in range(posx-opt.window_size//2,posx+opt.window_size//2+1):
#                             for j in range(posy-opt.window_size//2,posy+opt.window_size//2+1):
#                                 xi = i - posx + opt.window_size//2
#                                 yi = j - posy + opt.window_size//2
#                                 tmp += gaussian_weights[xi*opt.window_size+yi]*imgs_hr[b][c][i][j]
#                         uy_mssim[b][c].append(tmp.item())

#         for b in range(opt.batch_size):
#             for c in range(opt.channels):
#                 for posx in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                     for posy in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                         tmp = 0
#                         for i in range(posx-opt.window_size//2,posx+opt.window_size//2+1):
#                             for j in range(posy-opt.window_size//2,posy+opt.window_size//2+1):
#                                 xi = i - posx + opt.window_size//2
#                                 yi = j - posy + opt.window_size//2
#                                 tmp += gaussian_weights[xi*opt.window_size+yi]*(gen_h[b][c][i][j]-ux_mssim[b][c][(posx-opt.window_size//2)*opt.window_size+posy-opt.window_size//2])**2
#                         sigmax2_mssim[b][c].append(tmp.item())

#         for b in range(opt.batch_size):
#             for c in range(opt.channels):
#                 for posx in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                     for posy in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                         tmp = 0
#                         for i in range(posx-opt.window_size//2,posx+opt.window_size//2+1):
#                             for j in range(posy-opt.window_size//2,posy+opt.window_size//2+1):
#                                 xi = i - posx + opt.window_size//2
#                                 yi = j - posy + opt.window_size//2
#                                 tmp += gaussian_weights[xi*opt.window_size+yi]*(imgs_hr[b][c][i][j]-uy_mssim[b][c][(posx-opt.window_size//2)*opt.window_size+posy-opt.window_size//2])**2
#                         sigmay2_mssim[b][c].append(tmp.item())

#         for b in range(opt.batch_size):
#             for c in range(opt.channels):
#                 for posx in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                     for posy in range(opt.window_size//2,opt.hr_height-opt.window_size//2):
#                         tmp = 0
#                         for i in range(posx-opt.window_size//2,posx+opt.window_size//2+1):
#                             for j in range(posy-opt.window_size//2,posy+opt.window_size//2+1):
#                                 xi = i - posx + opt.window_size//2
#                                 yi = j - posy + opt.window_size//2
#                                 tmp += gaussian_weights[xi*opt.window_size+yi]*(gen_h[b][c][i][j]-ux_mssim[b][c][(posx-opt.window_size//2)*opt.window_size+posy-opt.window_size//2])*(imgs_hr[b][c][i][j]-uy_mssim[b][c][(posx-opt.window_size//2)*opt.window_size+posy-opt.window_size//2])
#                         sigmaxy_mssim[b][c].append(tmp.item())

#         mssim = 0
#         for b in range(opt.batch_size):
#             for c in range(opt.channels):
#                 for i in range((opt.hr_height-opt.window_size+1)**2):
#                     mssim[b][c][i] = (2 * ux_mssim[b][c][i]*uy_mssim[b][c][i] + c1) * (2 * sigmaxy_mssim[b][c][i] + c2) / ((ux_mssim[b][c][i] ** 2 + uy_mssim[b][c][i] ** 2 + c1) * (c2 + sigmax2_mssim[b][c][i]  + sigmay2_mssim[b][c][i] ))
#                     mssim += mssim[b][c][i]
#         mssim = mssim/(opt.batch_size*opt.channels*opt.window_size**2)


#         for wi in range(0,1):
#             print("wi:%d"%wi)
#             for wj in range(1):
#                 mat_mssim = []
#                 sigma_x2 = []
#                 for i in range(opt.window_size // 2, opt.window_size // 2 + opt.hr_height, 1):
#                     mat_line = []
#                     for j in range(opt.window_size // 2, opt.window_size// 2+opt.hr_width, 1):
#                     for j in range(opt.window_size // 2, opt.window_size// 2+opt.hr_width, 1):
#                         tmp = []
#                         x_bias = -(i-(opt.window_size//2))
#                         y_bias = -(j-(opt.window_size//2))
#                         for x in range(i - opt.window_size // 2, i + opt.window_size // 2, 1):
#                             tmp_line = []
#                             for y in range(j - opt.window_size // 2, j + opt.window_size // 2, 1):
#                                 tmp_line.append(gw[wi][wj][x+x_bias][y+y_bias] * (
#                                         gen_h_ex_numpy[wi][wj][x][y] - ux_mssim_numpy[wi][wj][x][y]) ** 2)
#                             tmp.append(tmp_line)

#                         tmp_sum = 0
#                         for ii in range(0,opt.window_size,1):
#                             for jj in range(0,opt.window_size,1):
#                                 tmp_sum += tmp[ii][jj]

#                         mat_line.append(tmp_sum)
#                     sigma_x2.append(mat_line)

#                 sigma_y2 = []
#                 # Franklin Delano Roosevelt
#                 for i in range(opt.window_size // 2, opt.window_size // 2 + opt.hr_height, 1):
#                     mat_line = []
#                     for j in range(opt.window_size // 2, opt.window_size // 2 + opt.hr_width, 1):
#                         tmp = []
#                         for x in range(i - opt.window_size // 2, i + opt.window_size // 2, 1):
#                             tmp_line = []
#                             for y in range(i - opt.window_size // 2, i + opt.window_size // 2, 1):
#                                 tmp_line.append(gw[x - i + opt.window_size // 2][y - i + opt.window_size // 2] * (
#                                         imgs_hr_ex_numpy[wi][wj][x][y] - uy_mssim_numpy[wi][wj][x][y]) ** 2)
#                             tmp.append(tmp_line)

#                         tmp_sum = 0
#                         for ii in range(0,opt.window_size-1,1):
#                             for jj in range(0,opt.window_size-1,1):
#                                 tmp_sum += tmp[ii][jj]

#                         mat_line.append(tmp_sum)

#                     sigma_y2.append(mat_line)

#                 sigma_xy = []
#                 for i in range(opt.window_size // 2, opt.window_size // 2 + opt.hr_height, 1):
#                     mat_line = []
#                     for j in range(opt.window_size // 2, opt.window_size // 2 + opt.hr_width, 1):
#                         tmp = []
#                         for x in range(i - opt.hr_height // 2, i + opt.hr_height // 2, 1):
#                             tmp_line = []
#                             for y in range(i - opt.hr_width // 2, i + opt.hr_width // 2, 1):
#                                 tmp_line.append(
#                                     gw[x - i + opt.hr_height // 2][y - i + opt.hr_width // 2] * (
#                                             imgs_hr_ex_numpy[wi][wj][x][y] - uy_mssim_numpy[wi][wj][x][y]) * (
#                                             gen_h_ex_numpy[wi][wj][x][y] - ux_mssim_numpy[wi][wj][x][y]))
#                             tmp.append(tmp_line)

#                         tmp_sum = 0
#                         for ii in range(tmp.size()):
#                             for jj in range(tmp[0].size()):
#                                 tmp_sum += tmp[ii][jj]

#                         mat_line.append(tmp_sum)
#                     sigma_xy.append(mat_line)

#                 tmp_ssim = []
#                 for i in range(sigma_xy.size()):
#                     tmp = []
#                     for j in range(sigma_xy[0].size()):
#                         tmp.append((2 * ux_mssim[wi][wj][i][j] + c1) * (2 * sigma_xy[i][j] + c2) / (
#                                 ux_mssim[wi][wj][i][j] ** 2 + uy_mssim[wi][wj][i][j] ** 2 + c1) * (
#                                            c2 + sigma_x2[wi][wj][i][j] ** 2 + sigma_y2[wi][wj][i][j] ** 2))
#                     tmp_ssim.append(tmp)

#                 mat_mssim.append(tmp_ssim.sum() / (opt.window_size ** 2))

#         mat_mssim = ((2*ux_mssim*uy_mssim+c1)*(2*sigmaxy_mssim+c2))/((ux_mssim2+uy_mssim2+c1)*(sigmax2_mssim+sigmay2_mssim+c2))
#         mssim = mat_mssim.sum() / (opt.batch_size * opt.channels*opt.hr_height*opt.hr_width
#     def srgan(self):
#         cuda = torch.cuda.is_available()
#         hr_shape = (self.opt.hr_height, self.opt.hr_width)
#         # test_shape = (128,128)

#         criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
#         criterion_content = torch.nn.L1Loss().to(device)
#         criterion_pixel = torch.nn.L1Loss().to(device)
#         criterion_cycle = torch.nn.L1Loss().to(device)
#         criterion_identity = torch.nn.L1Loss().to(device)
#         criterion_psnr = torch.nn.MSELoss().to(device)

#         # Initialize generator and discriminator
#         generator = GeneratorResNet()
#         discriminator = SRGANDiscriminator(input_shape = (self.opt.channels,*self.hr_shape))
#         feature_extractor = FeatureExtractor()

#         # Set feature extractor to inference mode
#         feature_extractor.eval()

#         # Losses
#         criterion_GAN = torch.nn.MSELoss()
#         criterion_content = torch.nn.L1Loss()  # 预测值与真实值绝对误差的平均数

#         if cuda:
#             generator = generator.cuda()
#             discriminator = discriminator.cuda()
#             feature_extractor = feature_extractor.cuda()
#             criterion_GAN = criterion_GAN.cuda()
#             criterion_content = criterion_content.cuda()

#         if self.opt.epoch != 0:
#             # Load pretrained models
#             generator.load_state_dict(torch.load("./saved_models/generator_%d.pth"))
#             discriminator.load_state_dict(torch.load("./saved_models/discriminator_%d.pth"))

#         # Optimizers adam使用SGD原理进行参数优化
#         optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
#         optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

#         # 不使用动量算法进行优化
#         # optimizer_G = torch.optim.RMSprop(generator.parameters(),lr=opt.lr,beta=(opt.b1,opt.b2))
#         # optimizer_D = torch.optim.RMSprop(discriminator.parameters(),lr=opt.lr,beta=(opt.b1,opt.b2))

#         Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#         dataloader = DataLoader(
#             ImageDataset("./%s" % self.opt.dataset_name, hr_shape=self.hr_shape),
#             batch_size = self.opt.batch_size,
#             shuffle = False,
#             num_workers= self.opt.n_cpu,
#         )


#         dataloadert = DataLoader(
#             ImageDataset("./%s" % self.opt.test_dataset_name,hr_shape = self.hr_shape),
#             batch_size = 1,
#             shuffle = False,
#             num_workers = self.opt.n_cpu,
#         )

#         dataloaderv = DataLoader(
#             ImageDataset("./%s" % self.opt.validation_dataset_name,hr_shape = self.hr_shape),
#             batch_size=1,
#             shuffle=False,
#             num_workers = self.opt.n_cpu
#         )

#         with open('./log/srgan/loss.txt','w') as file:
#             file.write('PIXEL_LOSS ')
#             file.write('CONTENT_LOSS ')
#             file.write('ADVERSIAL_LOSS ')
#             file.write('GENERATOR_LOSS ')
#             file.write('DISCRIMINATOR_LOSS ')
#             file.write('MSE ')
#             file.write('PSNR ')
#             file.write('SSIM ')
#             file.write('Mean SSIM')
#             file.write('\n')

#         # ----------
#         #  Training
#         # ----------

#         cnt = 0

#         for epoch in range(self.opt.epoch, self.opt.n_epochs):
#             for i, imgs in enumerate(dataloader):
#                 # Configure model input 把图像数据转换成tensor
#                 imgs_lr = Variable(imgs["lr"].type(Tensor))
#                 imgs_hr = Variable(imgs["hr"].type(Tensor))

#                 # Adversarial ground truths
#                 # requires_grad是参不参与误差反向传播,要不要计算梯度
#                 valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
#                 fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

#                 #pixel loss
#                 # ------------------
#                 #  Train Generators
#                 # ------------------

#                 optimizer_G.zero_grad()

#                 # Generate a high resolution image from low resolution input
#                 gen_hr = generator(imgs_lr)
#                 # test_hr = generator(imgs_test)

#                 loss_pixel_h = self.criterion_pixel(gen_hr, imgs_hr)


#                 # print(gen_hr.size())
#                 # Adversarial loss
#                 loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

#                 # Content loss
#                 gen_features = feature_extractor(gen_hr)
#                 real_features = feature_extractor(imgs_hr)
#                 loss_content = criterion_content(gen_features, real_features.detach())

#                 # # WGAN改进:每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
#                 # for parm in discriminator.parameters():
#                 #     parm.data.clamp_(-opt.clamp_num, opt.clamp_num)

#                 # Total loss
#                 loss_G = loss_content + 1e-3 * loss_GAN

#                 loss_G.backward()
#                 optimizer_G.step()

#                 # ---------------------
#                 #  Train Discriminator
#                 # ---------------------

#                 optimizer_D.zero_grad()

#                 # srgan的loss函数不含对数,不需要改进
#                 # Loss of real and fake images valid设置为1也就是高分辨率图像认为是1 fake设置为0也就是低分辨率图像设置为0
#                 loss_real = criterion_GAN(discriminator(imgs_hr), valid)
#                 loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

#                 # Total loss:这就是判别器损失
#                 loss_D = (loss_real + loss_fake) / 2

#                 # 反向传播
#                 loss_D.backward()
#                 optimizer_D.step()

#                 self.analyze(gen_hr,imgs_hr,self.opt,self.hr_shape,loss_pixel_h,loss_content,loss_GAN,loss_G,loss_D,'srgan')

#                 # --------------
#                 #  Log Progress
#                 # --------------

#                 sys.stdout.write(
#                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                     % (epoch, self.opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
#                 )

#                 batches_done = epoch * len(dataloader) + i

#                 if batches_done % self.opt.sample_interval == 0:
#                     # Save image grid with upsampled inputs and SRGAN outputs
#                     imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

#                     # print(" last2 gen_hr")
#                     # print(gen_hr.size())
#                     gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
#                     imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
#                     imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)

#                     # print("final--------\n")
#                     # print("imgs_lr")
#                     # print(imgs_lr.size())
#                     # print("imgs_hr")
#                     # print(imgs_hr.size())
#                     #
#                     # print("gen_hr")
#                     # print(gen_hr.size())
#                     # 值得注意的是左边一列是imgs_lr 右边一列是gen_hr 再右边一列我们把原高分辨图输出

#                     img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
#                     save_image(img_grid, "./images/srgan/train/%d.png" % batches_done, normalize=False)

#             for a,imgsv in enumerate(dataloaderv):
#                 imgsv_l = Variable(imgsv["lr"].type(Tensor))
#                 imgsv_h = Variable(imgsv["hr"].type(Tensor))
#                 genv_h = generator(imgsv_l)
#                 genv_h = make_grid(genv_h,nrow=1,noemalize=True)
#                 save_image(genv_h,"./validation/srgan/pic/%s" % str(epoch))
#                 ssim = pytorch_ssim.ssim(imgsv_h,genv_h)
#                 with open('./validation/srgan/text/%s' % str(epoch),'a') as file:
#                     file.write(ssim+'\n')

#             if self.opt.checkpoint_interval != -1 and epoch % self.opt.checkpoint_interval == 0:
#                 # Save model checkpoints
#                 torch.save(generator.state_dict(), "./saved_models/srgan_generator_%d.pth" % epoch)
#                 torch.save(discriminator.state_dict(), "./saved_models/srgan_discriminator_%d.pth" % epoch)


if __name__ == "__main__":
    mt = MainThread()
    mt.__init__()
