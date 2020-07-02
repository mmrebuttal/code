import torch,cv2
from torch import nn
from torchvision.models.vgg import vgg16
import numpy as np
from torch.autograd import Variable


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.VGG_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()
        self.fft_loss = fft_loss()
        self.bce_loss = nn.BCELoss()
        self.fitting_loss = FittingLoss()
        self.gabor_Loss = gabor_Loss()



    def forward(self, out_labels, out_images, ground_truth, texture):
        # Adversarial Loss
        real_lable = torch.FloatTensor(out_labels.size()).fill_(1.0)
        if torch.cuda.is_available():
            real_lable = real_lable.cuda()
        adversarial_loss = self.bce_loss(real_lable, out_labels.detach())
        # Perception Loss
        perception_loss = 0
        # perception_loss = self.mse_loss(self.VGG_network(out_images), self.VGG_network(ground_truth))
        # Image Loss
        image_loss = self.l1_loss(out_images, ground_truth)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # fft loss
        fft_loss = self.fft_loss(out_images, ground_truth)
        #total loss
        loss_all = image_loss + 0.01 * adversarial_loss + 0.06 * perception_loss + 2e-3 * tv_loss + 2e-5 * fft_loss + 0.06 * fft_loss
        return loss_all, image_loss, adversarial_loss, perception_loss, tv_loss, fft_loss

class fft_loss(nn.Module):
    def __init__(self):
        super(fft_loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, recon):
        input = input.unsqueeze(4)
        recon = recon.unsqueeze(4)
        input = torch.cat((input, input), 4)
        recon = torch.cat((recon, recon), 4)
        input_fft = torch.fft(input, 2)
        recon_fft = torch.fft(recon, 2)
        fft_loss = self.mse_loss(input_fft, recon_fft)
        return fft_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class fftLoss(nn.Module):
    def __init__(self, fit_loss_weight=1):
        super(fftLoss, self).__init__()
        self.fit_loss_weight = fit_loss_weight
        self.mse_loss = nn.MSELoss()

    def forward(self,  gt, rec, size, sigma1, sigma2):
        # tensor to numpy
        gt = gt.cpu().detach().numpy()
        rec = rec.cpu().detach().numpy()

        # high_fre_rec fft
        f_highrec = np.fft.fft2(rec)
        fshift_highrec = np.fft.fftshift(f_highrec)
        fimg_highrec = np.log(np.abs(fshift_highrec))

        # high_fre_gt fft
        f_highgt = np.fft.fft2(gt)
        fshift_highgt = np.fft.fftshift(f_highgt)
        fimg_highgt = np.log(np.abs(fshift_highgt))

        #gt fft
        f_gt = np.fft.fft2(gt)
        fshift_gt = np.fft.fftshift(f_gt)
        fimg_gt = np.log(np.abs(fshift_gt))
        fimg_gt = torch.from_numpy(fimg_gt)     #tensor

        # recontruct fft
        f_rec = np.fft.fft2(rec)
        fshift_rec = np.fft.fftshift(f_rec)
        fimg_rec = np.log(np.abs(fshift_rec))
        fimg_rec = torch.from_numpy(fimg_rec)   #tensor

        l2_fft = self.mse_loss(fimg_gt, fimg_rec)
        return l2_fft

