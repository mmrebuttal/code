import argparse
import os, torch, time, cv2
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from SKINET import *
from loss_skin import GeneratorLoss
from model import Generator, Discriminator
from image_loader import create_data_loader, create_test_loader
from utilize import zipDir,mkdir_path

# gen_filename_list("/data/landmark/npy", "npy_list.txt")
# gen_filename_list("/data/data_hr", "npy_list.txt")

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gauss_kernel', type=int, default=5, help='')

parser.add_argument('--with_possion', type=bool, default=False, help='')
parser.add_argument('--with_texturesyn_thesis', type=bool, default=False, help='')
parser.add_argument('--centralcropsize', type=int, default=800, help='')
parser.add_argument('--cropsize', type=int, default=256, help='')
parser.add_argument('--imgfilter_l', type=int, default=12, help='')
parser.add_argument('--imgfilter_h', type=int, default=20, help='')

# path
parser.add_argument("--device", default="cpu")
parser.add_argument("--train_filelist", default="/home/zhanghui07/srgan/train_list.txt")
parser.add_argument("--test_filelist", default="/home/zhanghui07/face_sr/src/hr_test.txt")
parser.add_argument("--val_filelist", default="/home/zhanghui07/face_sr/src/hr_val.txt")
parser.add_argument("--noise_path", default="/home/zhanghui07/face_sr/noise/")
parser.add_argument("--noise_1024_path", default="/home/zhanghui07/face_sr/noise_1024/")
parser.add_argument("--data_root", default="/data/data_hr/")

parser.add_argument("--save_path", default="/data/skin/")
parser.add_argument("--main_path", default="")
parser.add_argument("--gt_path", default="")
parser.add_argument("--in_path", default="")
parser.add_argument("--out_path", default="")
parser.add_argument("--recon_path", default="")
parser.add_argument("--test_path", default="")
parser.add_argument("--log_path", default="")
parser.add_argument("--model_path", default="")
parser.add_argument("--texture", default="")


if __name__ == '__main__':
    opt = parser.parse_args()
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    opt.main_path, main_filename = mkdir_path(opt,os.path.basename(__file__))
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    opt.GPU_IN_USE = torch.cuda.is_available()
    opt.device = torch.device('cuda' if opt.GPU_IN_USE else 'cpu')
    zipDir(os.getcwd(), opt.main_path+"code.zip")
    writer = SummaryWriter(opt.log_path)

    print('===> Loading datasets')
    train_loader = create_data_loader(opt) #create_old_loader(args)   #create_data_loader(args)
    val_loader = create_test_loader(opt)


    print('===> Construct network')
    netG = SKINET(in_channels=3, out_channels=3, nf=64, scale_factor=1).to(opt.device)     # netG = Generator(UPSCALE_FACTOR)
    netD = Discriminator()
    generator_criterion = GeneratorLoss()
    bce_loss = torch.nn.BCELoss()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    # print(netG)
    # print(netD)

    netG.weight_init(mean=0.0, std=0.02)

    # Multi-GPU support --
    if torch.cuda.device_count() > 1:
        print("Multiple GPU:", torch.cuda.device_count())
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        bce_loss.cuda()
    
    optimizerG = optim.Adam(netG.parameters(),lr=opt.lr)
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lr)
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    total_iter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        netG.train()
        netD.train()

        texture = torch.from_numpy(opt.texture)
        texture = texture.unsqueeze(0).permute(0, 3, 1, 2)

        for batch_num, dataset in enumerate(train_bar):
            img_lr = dataset['lr'].to(opt.device)    # low resolution
            img_hr = dataset['hr'].to(opt.device)    # high resolution
            filename = dataset["filename"]

            # cv2 to tensor: (B, H, W, C) -> (B, C, H, W)
            data = img_lr.permute(0, 3, 1, 2)
            target = img_hr.permute(0, 3, 1, 2)

            batch_size = img_lr.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            netD.zero_grad()
            real_img = Variable(target)
            data = Variable(data)

            if torch.cuda.is_available():
                real_img = real_img.cuda()
                data = data.cuda()

            fake_img = netG(data)
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()


            real_lable = torch.FloatTensor(real_out.size()).fill_(1.0)
            fake_lable = torch.FloatTensor(fake_out.size()).fill_(0.0)

            if torch.cuda.is_available():
                real_lable = real_lable.cuda()
                fake_lable = fake_lable.cuda()

            loss_fake = bce_loss(real_out, real_lable)
            loss_real = bce_loss(fake_out, fake_lable)

            d_loss = loss_fake + loss_real
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()

            fake_img = netG(data)
            fake_out = netD(fake_img).mean()

            g_loss, image_loss, adv_loss, perception_loss, tv_loss, fft_loss = generator_criterion(fake_out, fake_img, real_img, texture)
            g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

            print("===> Epoch[{}]({}/{}): g_Loss: {:.10f}, d_Loss: {:.10f}".format(epoch, batch_num, len(train_bar), g_loss, d_loss))

            if total_iter % 100 == 0:
                for i in range(fake_img.size(0)):
                    llr = data[i].cpu().permute(1, 2, 0).detach().numpy()               # low resolution face
                    hhr = target[i].cpu().permute(1, 2, 0).detach().numpy()             # high resolution face
                    rst = fake_img[i].cpu().permute(1, 2, 0).detach().numpy()

                    rst = cv2.cvtColor(rst, cv2.COLOR_RGB2BGR)
                    hhr = cv2.cvtColor(hhr, cv2.COLOR_RGB2BGR)
                    llr = cv2.cvtColor(llr, cv2.COLOR_RGB2BGR)

                    result2 = np.concatenate(
                        (llr * 255, rst * 255, hhr * 255, (rst - llr) * 20 * 255, (hhr - llr) * 20 * 255), axis=1)
                    cv2.imwrite(opt.recon_path + filename[0] + "_hr" + "_iter_" + str("%06d" % total_iter) + ".png",
                                result2)
            total_iter += 1

        netG.eval()
        # save model parameters
        torch.save(netG.state_dict(), opt.model_path +' netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), opt.model_path + 'netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))