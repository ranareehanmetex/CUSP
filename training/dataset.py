# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
from training.tools import getGaussianKernel
class AgeGauss:
    def __init__(self,k=41):
        self.k = torch.tensor(getGaussianKernel(k).T)[:,None].float()

    def __call__(self, ohe_age):
        if self.k.device != ohe_age.device:
            self.k = self.k.to(ohe_age.device)
        return torch.nn.functional.conv1d(ohe_age[:,None,:],self.k,padding=self.k.size(-1)//2)[:,0]

class AgeNumber:
    def __call__(self, ohe_age):
        return ohe_age.nonzero()[:,1][:,None].to(ohe_age.device)

class AgeBins:
    def __init__(self,bin_size=10):
        self.bin_size = bin_size

    def __call__(self, ohe_age):
        return torch.cat([torch.max(e, dim=1, keepdims=True)[0] for e in ohe_age.split(self.bin_size, dim=1)], dim=1)

class AgeDataset(ImageFolderDataset):
    # eval_labels_size = 6+1
    @staticmethod
    def get_eval_labels(labels, *args, **kwargs):
        tage = AgeDataset.__age_to_matrix__(np.ceil(np.linspace(20, 69, 6)).astype(int))
        eval_labels = np.concatenate([np.concatenate([a[None, :], tage], axis=0) for a in labels], axis=0)
        return eval_labels

    def __init__(self,
         age_np_path,
         image_path,
         use_labels = False,
         cmap_pre_kind = None,
         **super_kwargs
    ):
        super().__init__(path=image_path, use_labels=True, **super_kwargs)
        self.age_np_path = age_np_path
        self._image_fnames ,self._raw_labels = self._load_age_numpy()
        self._all_fnames = self._image_fnames
        self._raw_idx = np.arange(len(self._image_fnames))
        self._raw_shape[0] = len(self._image_fnames)

        if cmap_pre_kind == 'gauss':
            self.cmap_pre = AgeGauss()
            self.cmap_pre_dim = self.label_dim
        if cmap_pre_kind == 'number':
            self.cmap_pre = AgeNumber()
            self.cmap_pre_dim = 1
        if cmap_pre_kind == 'bins':
            bin_size = 10
            self.cmap_pre = AgeBins(bin_size)
            self.cmap_pre_dim = int(np.ceil(101/bin_size))



    # @property
    # def label_shape(self):
    #     return [0]
    @staticmethod
    def __age_to_matrix__(age):
        z = np.zeros((len(age), 101), dtype=np.float32)
        z[np.arange(len(age)), age] = 1
        return z

    def _load_age_numpy(self):
        age_np = np.load(self.age_np_path)
        file_sort = np.argsort(age_np[:, 0])
        _image_fnames, _raw_labels = age_np[file_sort].T
        # Age to float
        # _raw_labels = _raw_labels.astype(np.float32)
        _raw_labels = self.__age_to_matrix__(_raw_labels.astype(int))
        return _image_fnames, _raw_labels

    def _load_raw_labels(self):
        return self._load_age_numpy()[1]

import pandas as pd
import torchvision.transforms
import torchvision.transforms as T

class ToNumpy:
    def __call__(self,pic):
        return np.array(pic)

class VanillaToTensor:
    def __call__(self,pic):
        return torch.tensor(pic)

def get_transforms(dataset, is_train):
    if dataset == 'celeba':
        tr_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomHorizontalFlip'),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomRotation',
                            degrees = 20, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomCrop',
                            size=224),
            dnnlib.EasyDict(class_name='torchvision.transforms.ColorJitter',
                            brightness=.2, contrast=.5, saturation=.2, hue=0.02),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
        ts_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.CenterCrop',
                            size=224),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
    elif dataset == 'ffhq_lat':
        tr_transform = [
            dnnlib.EasyDict(class_name='torchvision.transforms.Resize', size=256),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomAffine',
                            degrees=20,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=10,
                            interpolation=T.InterpolationMode.BILINEAR),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomHorizontalFlip'),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
        ts_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.Resize', size=256),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
    elif dataset == 'bdd100k':
        ts_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.Resize', size=(252, 448)),
            dnnlib.EasyDict(class_name='torchvision.transforms.CenterCrop',
                            size=224),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
        tr_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.Resize',size=(252, 448)),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomHorizontalFlip'),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomRotation',
                            degrees=10, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            dnnlib.EasyDict(class_name='torchvision.transforms.ColorJitter',
                            brightness=.2, contrast=.5, saturation=.2, hue=0.02),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomCrop',
                            size=224),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
    elif dataset in ['afhq','zebra']:
        tr_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.Resize',size=224),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomHorizontalFlip'),
            dnnlib.EasyDict(class_name='torchvision.transforms.RandomAffine',
                            degrees=20,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=10,
                            interpolation=T.InterpolationMode.BILINEAR),
            # dnnlib.EasyDict(class_name='torchvision.transforms.ColorJitter',
            #                 brightness=.2, contrast=.5, saturation=.2, hue=0.1),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]
        ts_transform = [
            # dnnlib.EasyDict(class_name='training.dataset.VanillaToTensor'),
            dnnlib.EasyDict(class_name='torchvision.transforms.Resize',size=224),
            dnnlib.EasyDict(class_name='training.dataset.ToNumpy')
        ]


    else:
        raise NotImplementedError

    transform_list = tr_transform if is_train else ts_transform
    transform = T.Compose([
        dnnlib.util.construct_class_by_name(**d)
        for d in transform_list
    ])
    return transform

class ImageCSVDataset(ImageFolderDataset):
    @staticmethod
    def get_eval_labels(labels,random_generator,*args,**kwargs):
        eval_labels = np.concatenate([random_generator.get_eval_labels(l) for l in labels], axis=0)
        return eval_labels

    # @property
    # def eval_labels_size(self):
    #     return self[0][1].size + 1

    def __init__(self,
         csv_path,
         image_path,
         is_train,
         transforms = None,
         cmap_pre_kind = None,
         **super_kwargs
    ):
        if 'use_labels' in super_kwargs:
            del super_kwargs['use_labels']

        if transforms:
            self.transform = get_transforms(transforms, is_train)

        super().__init__(path=image_path, use_labels=True, **super_kwargs)

        self.csv_path = csv_path
        self.is_train = is_train
        self._image_fnames ,self._raw_labels = self._load_csv()
        self._all_fnames = self._image_fnames
        self._raw_idx = np.arange(len(self._image_fnames))
        self._raw_shape[0] = len(self._image_fnames)

        # self.resolution
        if cmap_pre_kind == 'number':
            self.cmap_pre = AgeNumber()
            self.cmap_pre_dim = 1



    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        # Get partition
        val_split = df.is_train == (1 if self.is_train else 0)
        df = df.loc[val_split]
        # Extract fnames
        _image_fnames = df.fname.to_numpy()
        # Extract labels
        _raw_labels = df.drop(columns=['fname','is_train']).to_numpy()
        # Age to float
        _raw_labels = _raw_labels.astype(np.float32)
        return _image_fnames, _raw_labels

    def _load_raw_image(self, raw_idx):
        # Load image
        fname = self._image_fnames[raw_idx]
        im = PIL.Image.open(os.path.join(self._path,fname))

        # Apply transform
        if self.transform:
            im = self.transform(im)
        im = np.array(im)

        # Convert to CHW
        if im.ndim == 2:
            im = im[:, :, np.newaxis] # HW => HWC
        im = im.transpose(2, 0, 1) # HWC => CHW

        return im

    def _load_raw_labels(self):
        return self._load_csv()[1]