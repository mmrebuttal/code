import pandas as pd
from PIL import Image
import numpy as np
import os,cv2, torch
from torch.utils import data
from torchvision import transforms
import data_generate

from data_generate import lr_data, get_noises_list
import pdb

class ImageDataset(data.Dataset):
    def __init__(self, opts):
        self.data_root = opts.data_root
        self.device =  opts.device
        self.main_path = opts.main_path
        hr_list = os.path.join(opts.train_filelist)
        self.datasets_hr = pd.read_csv(hr_list, header=None)
        self.noises_mean, self.noise_yuv, self.noises_weight = get_noises_list(opts.noise_path, 1)
        self.gauss_kernel = opts.gauss_kernel

        self.with_possion = opts.with_possion
        self.imgfilter_l = opts.imgfilter_l
        self.imgfilter_h = opts.imgfilter_h
        self.with_texturesyn_thesis = opts.with_texturesyn_thesis

        self.transform = transforms.Compose(
            [
                # transforms.Resize(256),
                transforms.CenterCrop(opts.centralcropsize),
                transforms.RandomCrop(opts.cropsize),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        filename_hr = os.path.join(self.data_root, self.datasets_hr.iloc[index, 0])
        # # PIL read
        image_hr = Image.open(filename_hr).convert('RGB')
        image_hr = self.transform(image_hr)
        image_hr = image_hr.permute(1, 2, 0)
        image_hr = image_hr *255

        # cv2 read
        # image_hr = cv2.imread(filename_hr)
        # image_hr = torch.from_numpy(image_hr)

        image_hr, image_lr, filename_lr = lr_data(self.imgfilter_l, self.imgfilter_h, self.with_possion, self.with_texturesyn_thesis, image_hr, filename_hr, self.noises_mean, self.noise_yuv, 1)

        image_hr = torch.clamp(image_hr/255, 0, 1)  #high resolution
        image_lr = torch.clamp(image_lr/255, 0, 1)  #low resolution

        ## add gauss blur
        hf = image_hr - torch.from_numpy(cv2.GaussianBlur(image_hr.numpy(), (self.gauss_kernel, self.gauss_kernel), 0))

        data = {'lr': image_lr, 'hr': image_hr, 'hf': hf, 'filename': filename_lr}

        # # test
        # hhr = image_hr.numpy()
        # hhr = cv2.cvtColor(hhr, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("./hr" + ".jpg", hhr*255 )
        # llr = image_lr.numpy()
        # llr = cv2.cvtColor(llr, cv2.COLOR_RGB2BGR)
        # cv2.imwrite( "./lr" + ".jpg", llr*255 )
        # hhf = hf.numpy()
        # hhf = cv2.cvtColor(hhf, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("./hf_5" + ".jpg", hhf * 255*10)


        return data

    def __len__(self):
        # assert len(self.datasets_hr) == len(self.datasets_lr)
        return len(self.datasets_hr)

class TestImageDataset(data.Dataset):
    def __init__(self, opts):
        self.data_root = opts.data_root
        self.device =  opts.device
        self.main_path = opts.main_path
        hr_list = os.path.join(opts.val_filelist)
        self.datasets_hr = pd.read_csv(hr_list, header=None)
        self.noises_mean, self.noise_yuv, self.noises_weight = get_noises_list(opts.noise_path, 1)

        self.with_possion = opts.with_possion
        self.imgfilter_l = opts.imgfilter_l
        self.imgfilter_h = opts.imgfilter_h
        self.with_texturesyn_thesis = opts.with_texturesyn_thesis

        self.transform = transforms.Compose(
            [
                # transforms.Resize(256),
                transforms.CenterCrop(opts.centralcropsize),
                transforms.RandomCrop(opts.cropsize),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        filename_hr = os.path.join(self.data_root, self.datasets_hr.iloc[index, 0])
        # # PIL read
        image_hr = Image.open(filename_hr).convert('RGB')
        image_hr = self.transform(image_hr)
        image_hr = image_hr.permute(1, 2, 0)
        image_hr = image_hr *255

        # cv2 read
        # image_hr = cv2.imread(filename_hr)
        # image_hr = torch.from_numpy(image_hr)
        # 0-255 cv2(256,256,3) (RGB)
        # image_hr, image_lr, filename_lr = lr_data(self.imgfilter_l, self.imgfilter_h, self.with_possion, self.with_texturesyn_thesis,image_hr, filename_hr, self.noises_mean, self.noise_yuv, 1)

        image_hr, image_lr, filename_lr = lr_data(self.imgfilter_l, self.imgfilter_h, self.with_possion,
                                                  self.with_texturesyn_thesis, image_hr, filename_hr, self.noises_mean,
                                                  self.noise_yuv, 1)

        image_hr = torch.clamp(image_hr/255, 0, 1)
        image_lr = torch.clamp(image_lr/255, 0, 1)
        data = {'lr': image_lr, 'hr': image_hr, 'filename': filename_lr}

        # RGB tensor (0,1)
        # # test
        # hhr = image_hr.numpy()
        # hhr = cv2.cvtColor(hhr, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(self.main_path + "hr" + ".jpg", hhr*255 )
        # llr = image_lr.numpy()
        # llr = cv2.cvtColor(llr, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(self.main_path + "lr" + ".jpg", llr*255 )

        return data

    def __len__(self):
        # assert len(self.datasets_hr) == len(self.datasets_lr)
        return len(self.datasets_hr)


def create_data_loader(opts):
    dataset = ImageDataset(opts)
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=opts.batchSize, shuffle=True, num_workers=opts.num_workers
    )
    return data_loader

def create_test_loader(opts):
    dataset = TestImageDataset(opts)
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=opts.batchSize, shuffle=True, num_workers=opts.num_workers
    )
    return data_loader

def create_old_loader(opts):
    dataset = ImageDataset_test(opts)
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=opts.batchSize, shuffle=True, num_workers=opts.num_workers
    )
    return data_loader

#----------old version below-------------------------------------------------------------------------------------------------------------#
class Image_test_Dataset(data.Dataset):
    def __init__(self, opts):
        self.data_root = opts.data_root
        self.device =  opts.device
        hr_list = os.path.join(opts.test_filelist)
        self.datasets_hr = pd.read_csv(hr_list, header=None)
        self.noises_mean, self.noise_yuv, self.noises_weight = get_noises_list(opts.noise_1024_path, 0.5)


    def __getitem__(self, index):
        filename_hr = os.path.join(self.data_root, self.datasets_hr.iloc[index, 0])
        image_hr = cv2.imread(filename_hr)
        image_hr = torch.from_numpy(image_hr)
        image_hr, image_lr, filename_lr = lr_data(image_hr, filename_hr, self.noises_mean, self.noise_yuv, 1)
        ## test
        # cv2.imwrite("./test/" + "hr" + ".jpg", image_hr)
        # cv2.imwrite("./test/" + "lr" + ".jpg", image_lr)
        # image_lr = image_lr.to(self.device)
        # image_hr = image_hr.to(self.device)

        data = {'lr': image_lr, 'hr': image_hr,'filename': filename_lr}
        # data = [image_lr, image_hr]
        return data

    def __len__(self):
        # assert len(self.datasets_hr) == len(self.datasets_lr)
        return len(self.datasets_hr)

class ImageDataset_test(data.Dataset):
    def __init__(self, opts):
        self.data_root = opts.data_root
        self.device =  opts.device
        hr_list = os.path.join(opts.test_filelist)
        self.datasets_hr = pd.read_csv(hr_list, header=None)
        self.noises_mean, self.noise_yuv, self.noises_weight = get_noises_list(opts.noise_path, 1)
        self.transform = transforms.Compose(
            [
                # transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        filename_hr = os.path.join(self.data_root, self.datasets_hr.iloc[index, 0])

        # # PIL read
        image_hr = Image.open(filename_hr).convert('RGB')
        image_hr = self.transform(image_hr)
        # noise_yuv = self.transform(self.noise_yuv)
        # noise_yuv = [self.transform(n) for n in self.noise_yuv]
        image_hr = image_hr.permute(1, 2, 0)
        image_hr = image_hr *255
        # noise_yuv = noise_yuv*255
        # cv2 read
        # image_hr = cv2.imread(filename_hr)
        # image_hr = torch.from_numpy(image_hr)

        image_hr, image_lr, filename_lr = lr_data(image_hr, filename_hr, self.noises_mean, self.noise_yuv, 1)
        ## test
        # cv2.imwrite(self.data_root + "hr" + ".jpg", image_hr)
        # cv2.imwrite("./test/" + "lr" + ".jpg", image_lr)

        # image_hr = image_hr.numpy()
        # image_lr = image_lr.numpy()
        # image_hr = Image.fromarray(cv2.cvtColor(image_hr, cv2.COLOR_BGR2RGB).astype(np.uint8))
        # image_lr = Image.fromarray(cv2.cvtColor(image_lr, cv2.COLOR_BGR2RGB).astype(np.uint8))
        # #
        # image_hr = self.transform(image_hr)
        # image_lr = self.transform(image_lr)
        # image_hr = image_hr.permute(1, 2, 0)
        # image_lr = image_lr.permute(1, 2, 0)

        # image_hr = torch.from_numpy(image_hr)
        # image_lr = torch.from_numpy(image_lr)
        data = {'lr': image_lr, 'hr': image_hr,'filename': filename_lr}
        return data

    def __len__(self):
        # assert len(self.datasets_hr) == len(self.datasets_lr)
        return len(self.datasets_hr)


class Image_trainold_Dataset(data.Dataset):
    def __init__(self, opts):
        self.data_root = opts.data_root
        self.device =  opts.device
        hr_list = os.path.join(opts.test_filelist)
        self.datasets_hr = pd.read_csv(hr_list, header=None)
        self.noises_mean, self.noise_yuv, self.noises_weight = get_noises_list(opts.noise_path, 0.5)


    def __getitem__(self, index):
        filename_hr = os.path.join(self.data_root, self.datasets_hr.iloc[index, 0])
        image_hr = cv2.imread(filename_hr)
        image_hr = torch.from_numpy(image_hr)
        image_hr, image_lr, filename_lr = lr_data(image_hr, filename_hr, self.noises_mean, self.noise_yuv, 1)
        ## test
        # cv2.imwrite("./test/" + "hr" + ".jpg", image_hr)
        # cv2.imwrite("./test/" + "lr" + ".jpg", image_lr)
        # image_lr = image_lr.to(self.device)
        # image_hr = image_hr.to(self.device)

        data = {'lr': image_lr, 'hr': image_hr,'filename': filename_lr}
        # data = [image_lr, image_hr]
        return data

    def __len__(self):
        # assert len(self.datasets_hr) == len(self.datasets_lr)
        return len(self.datasets_hr)




