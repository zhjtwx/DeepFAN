from __future__ import division
import imageio
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import config
import time
# from affine import affine_transform
import copy
# from torchvision import transforms
from rotation_3D import rotation3d, shear3d, rotation3d_itk
# from skimage.transform import resize
import json
import os
from utils.ran_brightness_contrast import RandomBrightnessContrast_corrected
from albumentations import Compose, RandomGamma
import cv2
import pandas as pd
import nibabel as nib
from enum import Enum
import tools
from torch.autograd import Variable
import torch.nn.functional as F

def json_load(path):
    """
    Load obj from json file
    """
    with open(path, 'r') as f:
        return json.load(f)
    return None


def load_img(filepath):
    img = imageio.imread(filepath)
    # img_shape = img.shape[1]
    # img_3d = img.reshape((-1, img_shape, img_shape))
    # img_3d = img_3d.transpose((1,2,0))
    return img


def load_img_path(directory):
    if directory.endswith('.png'):
        return load_img(directory)


def load_nii(load_fp):
    im = nib.load(str(load_fp))
    try:
        return np.asanyarray(im.dataobj), np.asanyarray(im.affine)
    except:
        return np.asanyarray(im.dataobj)


def load_nii_path(load_fp):
    img, _ = load_nii(load_fp)
    img_var = torch.from_numpy(img)
    img_var = Variable(torch.unsqueeze(torch.unsqueeze(img_var, 0), 0))
    img_var1 = F.interpolate(img_var, size=(156, 156, 156), mode='trilinear',
                            align_corners=True)  # 需要修改，原始是(32, 256, 256)
    img1 = torch.squeeze(img_var1).numpy()
    img1 = img1[np.newaxis, :, :, :]
    return img1


def do_crop(img, crop_size, crop_x, crop_y, crop_z):
    crop_xw = crop_x + crop_size[0]
    crop_yh = crop_y + crop_size[1]
    crop_zd = crop_z + crop_size[2]
    img = img[crop_z:crop_zd, crop_y: crop_yh, crop_x: crop_xw]
    return img


def constomized_RandomBrightnessContrast_aug(brightness_limit, contrast_limit, p=0.5):
    return Compose([
        RandomBrightnessContrast_corrected(brightness_limit=brightness_limit, contrast_limit=contrast_limit),
    ], p=p)


def constomized_RandomGamma_aug(gamma_limit, p=0.5):
    return Compose([
        RandomGamma(gamma_limit=gamma_limit),
    ], p=p)


class PixelWindow(Enum):
    NIL = (None, None)
    Lung = (-600, 1600)
    Bone = (400, 1600)
    Mediastinum = (40, 400)
    Aneurysm1 = (400, 1000)
    Artery1 = (400, 1200)
    LungMediastinum = (-580, 1640)


# 加窗宽窗位
def convert_window(image, window=PixelWindow.Lung, is_float=False, scale=255):
    window_center, window_width = window.value
    max_hu = window_center + window_width / 2
    min_hu = window_center - window_width / 2
    image_out = np.zeros(image.shape)
    w1 = (image > min_hu) & (image < max_hu)
    norm_to = float(scale)
    image_out[w1] = ((image[w1] - window_center + 0.5) / (window_width - 1.0) + 0.5) * norm_to
    image_out[image <= min_hu] = image[image <= min_hu] = 0.
    image_out[image >= max_hu] = image[image >= max_hu] = norm_to
    np_array = np.array(image_out)
    if is_float:
        return np_array
    np_array = np_array.astype('uint8')
    return np_array


class DatasetFromList(data.Dataset):
    def __init__(self, pair_image_list, roi_list, opt):
        super(DatasetFromList, self).__init__()
        self.seg_filenames = []
        self.max_wh_list = []

        self.label_pos = opt['label_pos']
        self.final_size = opt['final_size']
        self.shear = opt['shear']
        self.rotation = opt['rotation']
        self.train_crop = opt['train_crop']
        self.random_crop = opt['random_crop']
        self.flip = opt['flip']
        self.offset_max = opt['offset_max']
        self.ran_zoom = opt['ran_zoom']
        self.train_flag = opt['train_flag']
        self.pad = opt['pad']
        self.normalize = opt['normalize']
        self.test_zoom = opt['test_zoom']
        self.use_mask = opt['use_mask']
        self.pre_crop = opt['pre_crop']
        self.black_out = opt['black_out']
        self.random_brightness_limit = opt['random_brightness_limit']
        self.random_contrast_limit = opt['random_contrast_limit']
        self.black_in = opt['black_in']
        self.new_black_out = opt['new_black_out']
        self.new_black_in = opt['new_black_in']
        self.TBMSL_NET_opt = opt['TBMSL_NET_opt']
        self.use_mask_oneslice = opt['use_mask_oneslice']
        self.clahe = opt['clahe']
        if self.clahe:
            self.clahe_apply = cv2.createCLAHE(clipLimit=self.clahe[0], tileGridSize=self.clahe[1])
        if type(self.use_mask_oneslice) == str and self.use_mask_oneslice.endswith('.json'):
            self.json_use_mask_oneslice = json_load(self.use_mask_oneslice)

        if self.TBMSL_NET_opt:
            self.TBMSL_NET_opt_json = json_load(self.TBMSL_NET_opt['json'])

        if self.black_out:
            self.black_out_list = []
            self.black_out_dict = json_load(self.black_out['json'])
            self.black_out_cor = []

        if self.black_in:
            self.black_in_list = []
            self.black_in_dict = json_load(self.black_out)
            self.black_in_cor = []
        if self.random_brightness_limit or self.random_contrast_limit:
            if not self.random_brightness_limit:
                self.random_brightness_limit = [0, 0]
            if not self.random_contrast_limit:
                self.random_contrast_limit = [0, 0]
            self.constomized_RandomBrightnessContrast = constomized_RandomBrightnessContrast_aug(
                brightness_limit=self.random_brightness_limit,
                contrast_limit=self.random_contrast_limit)
        self.random_gamma_limit = opt['random_gamma_limit']
        if self.random_gamma_limit:
            self.constomized_RandomGamma = constomized_RandomGamma_aug(gamma_limit=self.random_gamma_limit)

        data_info = pd.read_csv(pair_image_list)  # 数据地址处理
        self.image_filenames = data_info['mask_img'].tolist()
        self.label_mb_list = data_info['label_mb'].tolist()
        self.label_lb_list = data_info['label_lb'].tolist()
        self.label_sp_list = data_info['label_sp'].tolist()
        self.label_dy_list = data_info['label_dy'].tolist()
        ###dataaug
        self.center_crop = opt['center_crop']
        self.scale = opt['scale']
        print('len_list', len(self.image_filenames))

    def __getitem__(self, index):
        input = load_nii_path(self.image_filenames[index])
        label_mb = self.label_mb_list[index]
        label_lb = self.label_lb_list[index]
        label_sp = self.label_sp_list[index]
        label_dy = self.label_dy_list[index]
        input = self.__data_aug(input)

        return torch.FloatTensor(input), np.array(label_mb), np.array(label_lb), \
               np.array(label_sp), np.array(label_dy), self.image_filenames[index], index

    def __len__(self):
        # print self.train_flag
        return len(self.image_filenames)

    def __data_aug(self, input):

        input_shape = input.shape[1]
        ############2D transform######################################
        if self.random_brightness_limit or self.random_contrast_limit:
            data = {"image": input, }
            data = self.constomized_RandomBrightnessContrast(**data)
            input = data['image']
        if self.random_gamma_limit:
            data = {"image": input, }
            data = self.constomized_RandomGamma(**data)
            input = data['image']
        input = input.reshape((-1, input_shape, input_shape))
        ori_input_shape = input.shape

        if self.pre_crop:
            pre_crop_x = np.shape(input)[2] // 2 - self.pre_crop[0] // 2
            pre_crop_y = np.shape(input)[1] // 2 - self.pre_crop[1] // 2
            pre_crop_z = np.shape(input)[0] // 2 - self.pre_crop[2] // 2
            input = do_crop(input, self.pre_crop, pre_crop_x, pre_crop_y, pre_crop_z)
            input = np.array(input)

        if self.ran_zoom and self.train_crop:
            ranzoom_x = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            ranzoom_y = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            ranzoom_z = np.random.uniform(low=self.ran_zoom[0], high=self.ran_zoom[1])
            zoom = [ranzoom_x, ranzoom_y, ranzoom_z]
        else:
            zoom = [1, 1, 1]

        if self.shear:
            hyx = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hzx = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hxy = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hzy = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hxz = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            hyz = (np.random.choice(2) * 2 - 1) * 0.2 * np.random.rand()
            shear = [hyx, hzx, hxy, hzy, hxz, hyz]
        else:
            shear = [0, 0, 0, 0, 0, 0]
        if self.pad:
            input = np.array(input)
            input = np.pad(input, self.pad, 'edge')
        if self.rotation == 'big':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(360)
        else:
            R_x, R_y, R_z = 0, 0, 0
        if self.test_zoom:
            zoom = self.test_zoom

        # if self.ran_zoom or self.rotation or self.shear or self.test_zoom:
        #     input = rotation3d_itk(input, R_x, R_y, R_z, zoom, shear)
        #     input = np.array(input)

        if self.train_crop:
            if self.random_crop:
                shift_x = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[0] + 1)
                shift_y = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[1] + 1)
                shift_z = (np.random.choice(2) * 2 - 1) * np.random.choice(self.random_crop[2] + 1)
            else:
                shift_x, shift_y, shift_z = 0, 0, 0

            crop_x = np.shape(input)[2] // 2 - self.train_crop[0] // 2 + shift_x  # + offset_w//2
            crop_y = np.shape(input)[1] // 2 - self.train_crop[1] // 2 + shift_y  # + offset_h//2
            crop_z = np.shape(input)[0] // 2 - self.train_crop[2] // 2 + shift_z  # + offset_d//2

            input = do_crop(input, self.train_crop, crop_x, crop_y, crop_z)
            input = np.array(input)

        if self.center_crop:
            crop_x = np.shape(input)[2] // 2 - self.center_crop[0] // 2
            crop_y = np.shape(input)[1] // 2 - self.center_crop[1] // 2
            crop_z = np.shape(input)[0] // 2 - self.center_crop[2] // 2
            input = do_crop(input, self.center_crop, crop_x, crop_y, crop_z)
            input = np.array(input)

        if self.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            input = input[::flip_z, ::flip_y, ::flip_x]

        mask_oneslice = []
        if self.clahe:
            input = input.reshape((-1, self.final_size[0]))  # self.final_size x y z
            input = self.clahe_apply.apply(input)
            input = input.reshape(self.final_size[::-1])

        input = input * self.scale
        input = torch.from_numpy(input)
        if self.normalize:
            input = (input - 0.5) / 0.5

        input = input.unsqueeze(0)
        if self.use_mask_oneslice:
            input = torch.cat((input, input * mask_oneslice, input * (1 - mask_oneslice)), dim=0)
        if self.use_mask:
            input = torch.cat((input, input * mask, input * (1 - mask)), dim=0)
        if 'bt_' in config.data_mode:
            input = input.transpose((2, 1, 0))
        return input

    def _get_random_params(self, offset_max, index=0):
        # np.random.seed(int(time.time() + 1e5 * index))

        rand_x = np.random.rand()
        rand_y = np.random.rand()
        rand_z = np.random.rand()
        rand_w = np.random.rand()
        rand_h = np.random.rand()
        rand_r = np.random.rand()
        rand_lr = np.random.rand()
        rand_td = np.random.rand()
        rand_r1 = np.random.rand()

        offset_x = int((rand_x * 2 - 1) * offset_max * 0.3)
        offset_y = int((rand_y * 2 - 1) * offset_max * 0.3)
        offset_z = int((rand_z * 2 - 1) * offset_max * 0.3)
        offset_w = int(((rand_w + 0.25) * 2 - 1) * offset_max * 2)
        offset_h = int(((rand_h + 0.25) * 2 - 1) * offset_max * 2)  # -10 30
        offset_d = int(((rand_h + 0.25) * 2 - 1) * offset_max * 2)
        # offset_w = int(((rand_w) * 2 - 1) * offset_max * 2)
        # offset_h = int(((rand_h) * 2 - 1) * offset_max * 2)
        rand_angle = int(rand_r * 360)

        return offset_x, offset_y, offset_z, offset_w, offset_h, offset_d, rand_angle, rand_lr, rand_td, rand_r1
