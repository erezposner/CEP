import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random



os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from img_utils.utils import TT_preprocess_multi_lvl_rgb
from dataloaders.dataloader import ToTensor
from utils import DistributedSamplerNoEvenlyDivisible

# max_max =  45000. / 255
MAX_DEPTH = 20
def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DemulatorDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DemulatorDataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DemulatorDataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DemulatorDataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DemulatorDataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
        self.new_size = (args.input_height, args.input_width)
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':

            rgb_file = sample_path.split()[0]
            depth_file = sample_path.split()[1]

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)

            image = Image.open(image_path).convert("RGB")
            # depth_gt = Image.open(depth_path)
            depth_gt = cv2.imread(depth_path,-1) / MAX_DEPTH#np.array(Image.open(depth_path)) / 255 / 256  # please use this to load ground truth depth during training and testing
            depth_gt = Image.fromarray(depth_gt)
            newsize = self.new_size
            image = image.resize(newsize)
            depth_gt = depth_gt.resize(newsize)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            

    
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            if self.args.do_dog is True:
                do_dog = random.random()
                if do_dog > 0.5:
                    image = (TT_preprocess_multi_lvl_rgb(image,scaling_sigs=2,alpha=10) / 255.0).astype(np.float32)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            # import matplotlib.pyplot as plt
            # plt.imshow(depth_gt, cmap=plt.get_cmap('gray_r'))
            # plt.show()
            depth_gt = np.expand_dims(depth_gt, axis=2)



            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(sample_path.split()[0])
            image = Image.open(image_path).convert("RGB")
            newsize =self.new_size
            image = image.resize(newsize)
            image = np.asarray(image, dtype=np.float32)/ 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])

                depth_path = os.path.join(gt_path, sample_path.split()[0].split('/')[0], sample_path.split()[1])

                has_valid_depth = False
                try:
                    depth_gt = cv2.imread(depth_path,
                                          -1) / MAX_DEPTH  # np.array(Image.open(depth_path)) / 255 / 256  # please use this to load ground truth depth during training and testing

                    # depth_gt = Image.open(depth_path)
                    # depth_gt = np.array(Image.open(
                    #     depth_path)) / 255 / 256  # please use this to load ground truth depth during training and testing
                    depth_gt = Image.fromarray(depth_gt) # please use this to load ground truth depth during training and testing

                    depth_gt = depth_gt.resize(newsize)

                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    # if self.args.dataset == 'nyu':
                    #     depth_gt = depth_gt / 1000.0
                    # else:
                    #     depth_gt = depth_gt / max_max

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            
            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        # do_flip = random.random()
        # if do_flip > 0.5:
        #     image = (image[:, ::-1, :]).copy()
        #     depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        # colors = np.random.uniform(0.9, 1.1, size=3)
        # white = np.ones((image.shape[0], image.shape[1]))
        # color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        # image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)



