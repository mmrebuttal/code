import nori2 as nori, pickle as pkl
import numpy as np, cv2, os, tqdm
import torch, time
import torch.nn.functional as F
from torch import distributions
from scipy.spatial import distance
from yuv import rgb_to_yuv, yuv_to_rgb

downsampling_scale = 0.5
noise_scale = 1


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = torch.from_numpy(img)
        # img = img/255
        filename, extension = os.path.splitext(filename)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

#1. noise
def noisy_np(noise_typ,image):
   if noise_typ == "gauss":
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,image.shape)
      gauss = gauss.reshape(image.shape).astype(np.float32)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.1
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals_ = len(np.unique(image)) #return none duplicate elements
      vals = 2 ** np.ceil(np.log2(vals_))
      noisy = np.random.poisson(image * vals)
      noisy = noisy / float(vals)
      # noisy = noisy - np.mean(noisy)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      mean = torch.zeros_like(image)
      vars = (0.1 ** 0.5) * torch.ones_like(image)
      gauss = torch.distributions.normal.Normal(mean, vars)
      noisy = gauss.sample() + image
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.1
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals_ = len(np.unique(image)) #return none duplicate elements
      vals = 2 ** np.ceil(np.log2(vals_))
      noisy = np.random.poisson(image * vals)
      noisy = noisy / float(vals)
      # noisy = noisy - np.mean(noisy)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


def add_possion(im):
    # im = torch.from_numpy(im)
    im = torch.distributions.poisson.Poisson(im)
    im = im.sample()
    return im
#2. blur
def denoisy(noise_typ,image):
   if noise_typ == "gauss":
      denoisy_img = cv2.GaussianBlur(image, (5, 5), 0)
      return denoisy_img
   elif noise_typ == "median":
       denoisy_img = cv2.medianBlur(image,5)
       return denoisy_img
   elif noise_typ =="bilateral":
       image = np.float32(image)
       denoisy_img = cv2.bilateralFilter(image,5,75,75)
       return denoisy_img

def bilateral(image, d, sigmagaussin, sigmagray):
    image = image.numpy()
    image = np.float32(image)
    denoisy_img = cv2.bilateralFilter(image, d, sigmagaussin, sigmagray)
    denoisy_img = torch.from_numpy(denoisy_img)
    return denoisy_img

#4. downsampling
def downsample(type, value, image):
    # image = torch.from_numpy(image)
    image = image.unsqueeze(0).permute(0, 3, 1, 2)
    if type == "nearest":
        image = F.interpolate(image, size=None, scale_factor=value, mode='nearest', align_corners=True)
        return image
    elif type == "bilinear":
        image = F.interpolate(image, size=None, scale_factor=value, mode='bilinear', align_corners=True)
        return image
    elif type == "bicubic":
        image = np.float32(image)
        image = F.interpolate(image, size=None, scale_factor=value, mode='bicubic', align_corners=True)
        return image
def upsample(type, value, image):
    if type == "nearest":
        image = F.interpolate(image, size=None, scale_factor=value, mode='nearest', align_corners=True)
        image = image.squeeze(0).permute(1, 2, 0)
        return image
    elif type == "bilinear":
        image = F.interpolate(image, size=value, scale_factor=None, mode='bilinear', align_corners=True)
        image = image.squeeze(0).permute(1, 2, 0)
        return image
    elif type == "bicubic":
        image = np.float32(image)
        image = F.interpolate(image, size=None, scale_factor=value, mode='bicubic', align_corners=True)
        image = image.squeeze(0).permute(1, 2, 0)
        return image

def get_noise_weight(img_value, noises_mean):
    dist = np.zeros(len(noises_mean))
    for k in range(len(noises_mean)):
        dst = distance.euclidean(img_value, noises_mean[k])
        dist[k] = dst
    dist_ = 1/dist
    weight_sum = np.sum(dist_)
    weight = dist_/weight_sum
    return weight


def img_filter(imgfilter_l, imgfilter_h,im):
    # 1.add noise

    im = add_possion(im)
    im = noisy("gauss", im)

    down_img = downsample('bilinear', downsampling_scale, im)
    up_img = upsample('bilinear', (im.size(0), im.size(1)), down_img)

    size = np.random.randint(imgfilter_l, imgfilter_h)
    sigma1 = np.random.randint(70, 90)
    sigma2 = np.random.randint(70, 90)
    denoisy_img = bilateral(up_img, size, sigma1, sigma2)

    # cv2.imwrite(LR_folder + filename + "_" + str(size)+ "_" +str(sigma) + ".jpg", denoisy_img)
    return denoisy_img, size, sigma1, sigma2

def img_filter_without_possion(imgfilter_l, imgfilter_h,im):
    # 1.add noise

    #im = add_possion(im)
    im = noisy("gauss", im)

    down_img = downsample('bilinear', downsampling_scale, im)
    up_img = upsample('bilinear', (im.size(0), im.size(1)), down_img)
    np.random.seed()
    size = np.random.randint(imgfilter_l, imgfilter_h)
    sigma1 = np.random.randint(70, 90)
    sigma2 = np.random.randint(70, 90)
    # print(size,sigma1,sigma2)
    denoisy_img = bilateral(up_img, size, sigma1, sigma2)
    return denoisy_img, size, sigma1, sigma2

def img_encode(im):
    return im

def get_noises_list(folder,weight_value):
    noises_mean = []
    noises_weight = []
    noises_yuv = []
    noises,filenames = load_images_from_folder(folder)
    weight_value = weight_value/(len(noises)-1)
    for noise in noises:
        noise_yuv = rgb_to_yuv(noise)
        noise_yuv_mean = torch.mean(noise_yuv[:,:,0])
        # print(np.mean(noise_yuv[:, :, 0] - noise_yuv_mean))
        noises_yuv.append(noise_yuv)
        noises_mean.append(noise_yuv_mean)
        noises_weight.append(weight_value)
    return noises_mean, noises_yuv, noises_weight

def reduce_light(im):
    img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    im_mean = img_yuv.mean()
    # low = im_mean*0.05
    # high = im_mean*0.05
    low = im_mean - im_mean * 0.5
    high = im_mean + im_mean * 0.5
    value = np.random.randint(low, high)
    img_yuv[:, :, 0] = img_yuv[:, :, 0] - value
    img_yuv[:, :, 0] = np.clip(img_yuv[:, :, 0], 0, 255).astype(np.float32)
    im_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    im_rgb = np.clip(im_rgb, 0, 255).astype(np.float32)
    # cv2.imwrite(LR_folder + filename + "_dark_" +str(value) + ".png", im_rgb)
    return im_rgb


def new_noise(imgfilter_l, imgfilter_h,with_possion,with_texturesyn_thesis,im, filename, noises_mean, noise_yuv, scale):
    if with_possion is True:
        im, diameter, sigma1, sigma2 = img_filter(imgfilter_l, imgfilter_h,im)
    elif with_possion is False:
        im, diameter, sigma1, sigma2 = img_filter_without_possion(imgfilter_l, imgfilter_h,im)
    if with_texturesyn_thesis is True:
        ###### add texture noise
        # convert to yuv
        img_yuv = rgb_to_yuv(im)
        #find nearest noise
        dist_img = torch.zeros((img_yuv.size(0), img_yuv.size(1), len(noises_mean)))
        for k in range(len(noises_mean)):
            distance = img_yuv[:, :, 0] - noises_mean[k] #
            dist_img[:, :, k] = torch.sqrt(torch.mul(distance, distance))

        # normalize
        dist_sums = torch.sum(dist_img,dim =2)
        noises_weight = 1.0 - torch.sigmoid(scale * (dist_img / dist_sums.unsqueeze(2)))

        # cal weighted noise
        for k in range(len(noises_mean)):
            img_yuv[:, :, 0] += noises_weight[:, :, k] * (noise_yuv[k][:, :, 0] - noises_mean[k])
        img_yuv[:, :, 0] = torch.clamp(img_yuv[:, :, 0], 0, 255)

        # convert to RBG
        im_rgb = yuv_to_rgb(img_yuv)
        ######
    elif with_texturesyn_thesis is False:
        im_rgb = im

    #filename
    filename_, extension = os.path.splitext(filename)
    filename = filename_.split('/', 3)[3].split('.', 1)[0]
    filename_lr = filename + "_" +str(diameter)+"_"+str(sigma1) +"_"+ str(sigma2)

    return im_rgb, filename_lr


def lr_data(imgfilter_l, imgfilter_h, with_possion,with_texturesyn_thesis, im, filename, noises_mean, noise_yuv, noise_scale):
    im = im.float()
    # im_dark_hr = reduce_light(im)
    im_noise_lr, filename_lr = new_noise(imgfilter_l, imgfilter_h, with_possion,with_texturesyn_thesis,im, filename, noises_mean, noise_yuv, noise_scale)
    # return im_dark_hr, im_noise_lr, filename_lr
    return im, im_noise_lr, filename_lr
