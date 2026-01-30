from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import random


import numpy as np

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl, self.task)

            
            if self.opt['phase'] == 'train':

                self.lq_2 = "/home/dataset/SODA_DRealSR/IMG/Train/LR"
                self.gt_2 = "/home/dataset/SODA_DRealSR/IMG/Train/HR"
                self.path2 = paired_paths_from_folder([self.lq_2, self.gt_2], ['lq', 'gt'], self.filename_tmpl, self.task)
                self.leng2 = len(self.path2)

                self.lq_3 = "/home/dataset/SODA_DRealSR/Canon/Train/LR"
                self.gt_3 = "/home/dataset/SODA_DRealSR/Canon/Train/HR"
                self.path3 = paired_paths_from_folder([self.lq_3, self.gt_3], ['lq', 'gt'], self.filename_tmpl, self.task)
                self.leng3 = len(self.path3)

                # self.lq_4 = "/home/tangzz/Dataset/SODA_DRealSR/panasonic/Train/LR"
                # self.gt_4 = "/home/tangzz/Dataset/SODA_DRealSR/panasonic/Train/HR"
                # self.path4 = paired_paths_from_folder([self.lq_3, self.gt_3], ['lq', 'gt'], self.filename_tmpl, self.task)
                # self.leng4 = len(self.path3)

                # self.lq_5 = "/home/tangzz/Dataset/SODA_DRealSR/sony/Train/LR"
                # self.gt_5 = "/home/tangzz/Dataset/SODA_DRealSR/sony/Train/HR"
                # self.path5 = paired_paths_from_folder([self.lq_3, self.gt_3], ['lq', 'gt'], self.filename_tmpl, self.task)
                # self.leng5 = len(self.path3)

                # self.lq_6 = "/home/tangzz/Dataset/SODA_DRealSR/DSC/Train/LR"
                # self.gt_6 = "/home/tangzz/Dataset/SODA_DRealSR/DSC/Train/HR"
                # self.path6 = paired_paths_from_folder([self.lq_3, self.gt_3], ['lq', 'gt'], self.filename_tmpl, self.task)
                # self.leng6 = len(self.path3)

                self.Path_all = {"1": self.paths, "2": self.path2, "3":self.path3}
                self.Num_all = {"1": len(self.paths), "2": self.leng2, "3":self.leng3}




    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1

            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
    
        elif self.task == 'denoising_gray': # Matlab + OpenCV version
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            # OpenCV version, following "Deep Convolutional Dictionary Learning for Image Denoising"
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            # # Matlab version (using this version may have 0.6dB improvement, which is unfair for comparison)
            # img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            # if img_gt.ndim != 2:
            #     img_gt = rgb2ycbcr(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), y_only=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)

        else:   
            
            if self.opt['phase'] == 'train':

                options = ["1", "2", "3"]

                sele = random.choice(options)

                Selected_path = self.Path_all[sele]
                Selected_num = self.Num_all[sele]

                s_index = random.randint(0, Selected_num-1)

                gt_path = Selected_path[s_index]['gt_path']
                lq_path = Selected_path[s_index]['lq_path']
                
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)

                img_bytes = self.file_client.get(lq_path, 'lq')
                img_lq = imfrombytes(img_bytes, float32=True)

                Select_index = int(sele) - 1  #用来指出当前图片属于哪一个域

            else:
                
                #测试时，就仅使用当前样本
                gt_path = self.paths[index]['gt_path']
                lq_path = self.paths[index]['lq_path']
                
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)

                img_bytes = self.file_client.get(lq_path, 'lq')
                img_lq = imfrombytes(img_bytes, float32=True)


                if "LR/P" in lq_path:
                    Select_index = 0
                elif "LR/IMG" in lq_path:
                    Select_index = 1
                elif "LR/Canon" in lq_path:
                    Select_index = 2
                else:
                    Select_index = 1000



                







        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if self.opt['phase'] == 'train':
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'Class_gt': Select_index}
        else:
              return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'Class_gt': Select_index}          

    def __len__(self):
        return len(self.paths)
