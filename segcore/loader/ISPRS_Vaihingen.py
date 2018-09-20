import os
import random
import torch
import numpy as np
from torch.utils import data
from skimage import io

class ISPRSVaihingenLoader(data.Dataset):
    def __init__(self, cfg=None, training=True):
        super(ISPRSVaihingenLoader, self).__init__()

        base_datapath = cfg["data"]["path"]
        train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
        test_ids = ['5', '21', '15', '30']
        data_files = base_datapath + 'ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area{}.tif'
        label_files = base_datapath + 'ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area{}.tif'
        eroded_files = base_datapath + 'ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
        cache = True
        augmentation = False

        self.training = training
        self.augmentation = augmentation
        self.cache = cache
        self.window_size = (256, 256)

        # list of files
        if self.training:
            self.data_files = [data_files.format(id) for id in train_ids]
            self.label_files = [label_files.format(id) for id in train_ids]
            check_files = self.data_files + self.label_files
        else:
            self.data_files = [data_files.format(id) for id in test_ids]
            self.label_files = [label_files.format(id) for id in test_ids]
            self.eroded_files = [eroded_files.format(id) for id in test_ids]
            check_files = self.data_files + self.label_files + self.eroded_files

        # if some files do not exist give an error remind
        for f in check_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file!'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

        # ISPRS color palette
        # Let's define the standard ISPRS color palette
        self.palette = {0: (255, 255, 255),  # Impervious surfaces (white)
                        1: (0, 0, 255),  # Buildings (blue)
                        2: (0, 255, 255),  # Low vegetation (cyan)
                        3: (0, 255, 0),  # Trees (green)
                        4: (255, 255, 0),  # Cars (yellow)
                        5: (255, 0, 0),  # Clutter (red)
                        6: (0, 0, 0)}  # Undefined (black)

        self.invert_palette = {v: k for k, v in self.palette.items()}

    def __len__(self):
        # Default epoch size is 10000 sample
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays):
        will_flip, will_mirror = False, False
        if random.random() < 0.5:
            will_flip = True
        if random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def convert_to_color(self, arr_2d):
        """ Numeric labels to RGB-color encoding """
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in self.palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d

    def convert_from_color(self, arr_3d):
        """ RGB-color encoding to grayscale labels """
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in self.invert_palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d

    def get_random_pos(self, img, window_shape):
        """ Extract of 2D random patch of shape window_shape in the image """
        w, h = window_shape
        W, H = img.shape[-2:]
        x1 = random.randint(0, W - w - 1)
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1)
        y2 = y1 + h
        return x1, x2, y1, y2


    def __getitem__(self, index):

        if self.training:
            # pick a random image
            random_idx = random.randint(0, len(self.data_files) - 1)

            # if the title hasn't been loaded yet, put in cache
            if random_idx in self.data_cache_.keys():
                data = self.data_cache_[random_idx]
            else:
                # Data is normalized in [0, 1]
                data = 1.0 / 255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2, 0, 1)), dtype='float32')
                if self.cache:
                    self.data_cache_[random_idx] = data

            if random_idx in self.label_cache_.keys():
                label = self.label_cache_[random_idx]
            else:
                # Labels are converted from RGB to their numeric values
                label = np.asarray(self.convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
                if self.cache:
                    self.label_cache_[random_idx] = label

            # Get a random patch
            x1, x2, y1, y2 = self.get_random_pos(data, self.window_size)
            data_p = data[:, x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]

            # Data augmentation
            data_p, label_p = self.data_augmentation(data_p, label_p)

            # Return the torch.Tensor values
            return (torch.from_numpy(data_p),
                    torch.from_numpy(label_p))

        else:
            val_images = 1.0 / 255 * np.asarray(io.imread(self.data_files[index]), dtype='float32')
            val_labels = np.asarray(io.imread(self.label_files[index]), dtype='uint8')
            eroded_labels = self.convert_from_color(io.imread(self.eroded_files[index]))

            return (val_images, val_labels, eroded_labels)

