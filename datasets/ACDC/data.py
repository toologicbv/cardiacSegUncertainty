# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 17:25:20 2017

@author: BobD
"""

import os
import glob
from os import path
import numpy as np
from scipy.ndimage import interpolation
from tqdm import tqdm
import scipy
import SimpleITK as sitk
import nibabel as nib
from datasets.common import apply_2d_zoom_3d
from torch.utils.data import Dataset


def basename(arg):
    try:
        return os.path.splitext(os.path.basename(arg))[0] # is raising an exception faster than an if clause?
    except Exception as e:
        if isinstance(arg, list):
            return [basename(el) for el in arg]
        else:
            raise e

def saveImage(fname, arr, spacing):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname, False)


def circle_mask(imsize, radius=1.):
    #    imsize = 480
    xx, yy = np.mgrid[:imsize, :imsize]
    circle = (xx - imsize / 2) ** 2 + (yy - imsize / 2) ** 2
    return circle < (radius * imsize / 2) ** 2


def sitk_save(fname, arr, spacing=None, dtype=np.int16):
    if type(spacing) == type(None):
        spacing = np.ones((len(arr.shape),))
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname, True)

saveMHD = sitk_save

def sitk_read(fname):
    img = sitk.ReadImage(fname)
    spacing = img.GetSpacing()[::-1]
    im = sitk.GetArrayFromImage(img)
    return im, spacing

readMHD = sitk_read

def readMHDInfo(mhd_file):
    with open(mhd_file, 'r') as f:
        for l in f:
            if l.startswith('ElementSpacing'):
                spacing = np.array(l[17:].split()[::-1], np.float)
            if l.startswith('DimSize'):
                dims = np.array(l[10:].split()[::-1], np.uint)
    return dims, spacing


def extract_planes(data):
    shape = data.shape
    centers = np.divide(shape, 2).astype(int)
    return data[centers[0]], data[:, centers[1]], data[:, :, centers[2]]


def expand_zaxis_resolution(img, z_spacing=2, as_type=np.float32):
    # Resample 3D image in through plane direction (z-axis) if original spacing in that direction is larger than new z-spacing
    orig_spacing = img.GetSpacing()
    if orig_spacing[2] > z_spacing:
        img_expander = sitk.ExpandImageFilter()
        if as_type == np.float32:
            img_expander.SetInterpolator(sitk.sitkLanczosWindowedSinc)
        elif as_type == np.int:
            img_expander.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            raise ValueError("ERROR - expand_zaxis_resolution - type unknown {}".format(as_type))

        expand_factor = int(orig_spacing[2] / z_spacing)  # original spacing z-axis / 2mm
        img_expander.SetExpandFactors((1, 1, expand_factor))
        return img_expander.Execute(img)
    else:
        return img


def read_nifty(fname, resample_zaxis=False, as_type=np.float32):

    img = sitk.ReadImage(fname)
    if resample_zaxis:
        img = expand_zaxis_resolution(img, z_spacing=5, as_type=as_type)

    spacing = img.GetSpacing()[::-1]
    arr = sitk.GetArrayFromImage(img)
    if resample_zaxis and as_type == np.int:
        # reference labels. We make sure the label values are rounded
        arr = np.round(arr).astype(as_type)

    return arr, spacing


def apply_2d_zoom(arr4d, spacing):
    vox_size = 1.4  # mm
    zoom = np.array(spacing, float)[1:] / vox_size
    for idx in range(arr4d.shape[0]):
        for jdx in range(arr4d.shape[1]):
            sigma = .25 / zoom
            arr4d[idx, jdx] = scipy.ndimage.gaussian_filter(arr4d[idx, jdx], sigma)
    return scipy.ndimage.interpolation.zoom(arr4d, (1, 1) + tuple(zoom), order=1), np.array(
        (spacing[0], vox_size, vox_size), np.float32)


def rescale_intensities(im, dtype=np.float32):
    min_val, max_val = np.percentile(im, (1, 99))
    im = ((im.astype(dtype) - min_val) / (max_val - min_val)).clip(0, 1)
    return im


def get_acdc_patient_ids(fold, dataset, limited_load=False):
    assert dataset in ['training', 'validation', 'full']
    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)
    training_nums, validation_nums = allpatnumbers[foldmask != fold], allpatnumbers[foldmask == fold]
    if dataset == 'training':
        pat_nums = training_nums
    elif dataset == 'validation':
        pat_nums = validation_nums
    elif dataset == 'full':
        pat_nums = np.arange(1, 101)
    if limited_load:
        pat_nums = pat_nums[:15]
    return pat_nums


def acdc_validation_fold(fold, root_dir=None, limited_load=False, resample=False, patid=None, resample_zaxis=False):
    # patid --> should be integer, so 21 instead of patient021
    rescale = True
    if isinstance(patid, str):
        patid = int(patid.strip('patient'))

    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)
    validation_nums = allpatnumbers[foldmask == fold]
    if patid is not None:
        validation_nums = [pid for pid in validation_nums if patid == pid]
        if len(validation_nums) == 0:
            raise ValueError("ERROR - acdc validation fold - patid {} not found in validation set".format(patid))
    if limited_load:
        validation_nums = validation_nums[:3]

    for patnum in validation_nums:
        img = ACDCImage(patnum, root_dir=root_dir, resample=resample, scale_intensities=rescale, resample_zaxis=resample_zaxis)
        ed, sp, ed_gt = img.ed()
        es, _, es_gt = img.es()
        frame_id_ed, frame_id_es = img.get_frame_id('ED'), img.get_frame_id('ES')

        yield {'image': ed, 'spacing': sp, 'reference': ed_gt, 'patient_id': img.patient_id, 'frame_id': frame_id_ed,
               'cardiac_phase': 'ED', 'structures': [], 'original_spacing': img.original_spacing}
        yield {'image': es, 'spacing': sp, 'reference': es_gt, 'patient_id': img.patient_id, 'frame_id': frame_id_es,
               'cardiac_phase': 'ES', 'structures': [], 'original_spacing': img.original_spacing}


def acdc_get_all_patients(root_dir=None, limited_load=False, resample=False, patid=None, resample_zaxis=False):
    # patid --> should be integer, so 21 instead of patient021
    rescale = True
    if isinstance(patid, str):
        patid = int(patid.strip('patient'))

    allpatnumbers = np.arange(1, 101)
    if patid is not None:
        allpatnumbers = [pid for pid in allpatnumbers if patid == pid]
        if len(allpatnumbers) == 0:
            raise ValueError("ERROR - acdc_get_all_patients - patid {} does not exist in dataset".format(patid))
    if limited_load:
        allpatnumbers = allpatnumbers[:3]

    for patnum in allpatnumbers:
        img = ACDCImage(patnum, root_dir=root_dir, resample=resample, scale_intensities=rescale, resample_zaxis=resample_zaxis)
        ed, sp, ed_gt = img.ed()
        es, _, es_gt = img.es()
        frame_id_ed, frame_id_es = img.get_frame_id('ED'), img.get_frame_id('ES')

        yield {'image': ed, 'spacing': sp, 'reference': ed_gt, 'patient_id': img.patient_id, 'frame_id': frame_id_ed,
               'cardiac_phase': 'ED', 'structures': [], 'original_spacing': img.original_spacing}
        yield {'image': es, 'spacing': sp, 'reference': es_gt, 'patient_id': img.patient_id, 'frame_id': frame_id_es,
               'cardiac_phase': 'ES', 'structures': [], 'original_spacing': img.original_spacing}


def acdc_validation_fold_image4d(fold, root_dir, patid=None, resample=False, rescale=False, file_suffix='4d.nii.gz'):
    if isinstance(patid, str):
        patid = int(patid.strip('patient'))

    print("acdc_validation_fold_image4d ", fold)
    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)
    validation_nums = allpatnumbers[foldmask == fold]
    if patid is not None:
        validation_nums = [pid for pid in validation_nums if patid == pid]
        if len(validation_nums) == 0:
            raise ValueError("ERROR - acdc validation fold - patid {} not found in validation set".format(patid))

    for patnum in validation_nums:
        img = ACDCImage(patnum, root_dir=root_dir, resample=True, scale_intensities=False, resample_zaxis=False,
                        file_suffix=file_suffix)
        img4d_arr = img.image4d()
        num_of_frames = img4d_arr.shape[0]
        spacing = img.voxel_spacing()[1:]
        original_spacing = img.voxel_spacing()[1:]

        for frame_id in range(num_of_frames):
            img_np = img4d_arr[frame_id]
            if resample:
                print("WARNING - Resampling")
                spacing = np.array([original_spacing[0], ACDCImage.new_spacing[1], ACDCImage.new_spacing[2]]).astype(np.float32)
                img_np = apply_2d_zoom_3d(img_np, spacing, do_blur=True, new_spacing=img.new_spacing)
            if rescale:
                img_np = rescale_intensities(img_np).astype(np.float32)

            yield {'image': img_np, 'spacing': spacing, 'reference': img_np, 'patient_id': img.patient_id, 'frame_id': frame_id,
                   'cardiac_phase': 'ED', 'structures': [], 'original_spacing': original_spacing, 'num_of_frames': num_of_frames}


def get_4dimages_nifti(data_src_dir, patid=None, resample=False, rescale=False, file_suffix=".nii.gz"):

    if patid is None:
        search_path = os.path.join(data_src_dir, "*{}".format(file_suffix))
    else:
        search_path = os.path.join(data_src_dir, patid + "*{}".format(file_suffix))
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_path))
    else:
        print("INFO - Found {} nifti files in {}".format(len(file_list), data_src_dir))
    file_list.sort()

    for fname in file_list:
        img = sitk.ReadImage(fname)
        img4d_arr = sitk.GetArrayFromImage(img).astype(np.float32)
        pat_id, _ = (os.path.splitext(os.path.basename(fname))[0]).split(".")
        num_of_frames = img4d_arr.shape[0]
        print("INFO - Processing patid {}, #frames {}".format(pat_id, num_of_frames))
        spacing = img.GetSpacing()[::-1][1:]
        original_spacing = img.GetSpacing()[::-1][1:]
        if resample or spacing[-1] < 1.:
            do_resample = True
            new_spacing = np.array([original_spacing[0], 1.4, 1.4]).astype(np.float32)
            spacing = new_spacing
        else:
            do_resample = False
            new_spacing = None

        for frame_id in range(num_of_frames):
            img_np = img4d_arr[frame_id]
            if do_resample:
                img_np = apply_2d_zoom_3d(img_np, original_spacing, new_spacing, do_blur=True)
            if rescale:
                img_np = rescale_intensities(img_np).astype(np.float32)

            yield {'image': img_np, 'spacing': spacing, 'reference': img_np, 'patient_id': pat_id, 'frame_id': frame_id,
                   'cardiac_phase': 'ED', 'structures': [], 'original_spacing': original_spacing, 'num_of_frames': num_of_frames}


def get_images_nifti(data_src_dir, patid=None, resample=False, rescale=False):

    if patid is None:
        search_path = os.path.join(data_src_dir, "*.nii.gz")
    else:
        search_path = os.path.join(data_src_dir, patid + "*.nii.gz")
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_path))
    else:
        print("INFO - Found {} nifti files in {}".format(len(file_list), data_src_dir))
    file_list.sort()

    for fname in file_list:
        img = sitk.ReadImage(fname)
        img_arr = sitk.GetArrayFromImage(img).astype(np.float32)
        pat_id, _ = (os.path.splitext(os.path.basename(fname))[0]).split(".")
        if img_arr.ndim == 4:
            num_of_frames = img_arr.shape[0]
            spacing = img.GetSpacing()[::-1][1:]
            original_spacing = img.GetSpacing()[::-1][1:]
        else:
            num_of_frames = 1
            img_arr = np.expand_dims(img_arr, axis=0)
            spacing = img.GetSpacing()[::-1]
            original_spacing = img.GetSpacing()[::-1]
        print("INFO - Processing patid {}, #frames {}".format(pat_id, num_of_frames))
        if resample or spacing[-1] < 1.:
            do_resample = True
            new_spacing = np.array([original_spacing[0], 1.4, 1.4]).astype(np.float32)
            spacing = new_spacing
        else:
            do_resample = False
            new_spacing = None

        for frame_id in range(num_of_frames):
            img_np = img_arr[frame_id]
            if do_resample:
                img_np = apply_2d_zoom_3d(img_np, original_spacing, new_spacing, do_blur=True)
            if rescale:
                img_np = rescale_intensities(img_np).astype(np.float32)

            yield {'image': img_np, 'spacing': spacing, 'reference': img_np, 'patient_id': pat_id, 'frame_id': frame_id,
                   'cardiac_phase': 'none', 'structures': [], 'original_spacing': original_spacing, 'num_of_frames': num_of_frames}


class ACDCImage(object):
    new_spacing = tuple((1., 1.4, 1.4))

    def __init__(self, number, root_dir='~/data/ACDC/all_cardiac_phases', scale_intensities=True, resample=False,
                 resample_zaxis=False, file_suffix="4d.nii.gz"):
        # number IS patient id without "patient" prefix, just integer without lpad zeros
        self._number = number
        self._patient_id = "patient{:03d}".format(number)
        self._path = root_dir + '/patient{:03d}'.format(number)
        self._img4d_fname = self._path + '/patient{:03d}_{}'.format(self._number, file_suffix)
        self._img4d = None
        self.number_of_frames = None
        self._scale_intensities = scale_intensities
        # IMPORTANT: resampling of patient029 ends up in different shape then before. Hence, for this patient we do not resample
        if self._number == 29:
            self._resample = False
        else:
            self._resample = resample
        self._resample_zaxis = resample_zaxis
        self.spacing = None
        self.original_spacing = None
        self.base_apex_slice_es = None
        self.base_apex_slice_ed = None
        self.frame_id = None

    def ed(self):
        idx = int(self.info()['ED'])
        self.frame_id = idx
        im, self.original_spacing = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number, idx)),
                                               self._resample_zaxis)
        self.original_spacing = self._check_spacing(self.original_spacing)
        gt, _ = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}_gt.nii.gz'.format(self._number, idx)),
                           resample_zaxis=self._resample_zaxis, as_type=np.int)
        self.spacing = self.original_spacing
        if self._scale_intensities:
            im = rescale_intensities(im).astype(np.float32)

        if self._resample or self.original_spacing[-1] < 1.:
            im, gt = self._do_resample(im, gt, self.original_spacing)
            self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)

        self.base_apex_slice_ed = self._determine_apex_base_slices(gt)
        return im, self.spacing, gt

    def es(self):
        idx = int(self.info()['ES'])
        self.frame_id = idx
        im, self.original_spacing = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number, idx)),
                                               self._resample_zaxis)
        self.original_spacing = self._check_spacing(self.original_spacing)
        gt, _ = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}_gt.nii.gz'.format(self._number, idx)),
                           resample_zaxis=self._resample_zaxis, as_type=np.int)
        self.spacing = self.original_spacing if isinstance(self.original_spacing, np.ndarray) else \
            np.array(self.original_spacing)
        if self._scale_intensities:
            im = rescale_intensities(im).astype(np.float32)
        if self._resample or self.original_spacing[-1] < 1.:
            im, gt = self._do_resample(im, gt, self.original_spacing)
            self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
        self.base_apex_slice_es = self._determine_apex_base_slices(gt)
        return im, self.spacing, gt

    def get(self, cardiac_phase):
        if cardiac_phase == "ES":
            im, self.spacing, gt = self.es()
            return im, self.spacing, gt, self.frame_id
        else:
            im, self.spacing, gt = self.ed()
            return im, self.spacing, gt, self.frame_id

    def all(self):
        return self._return_all_phases()

    def _return_all_phases(self):
        # Use this method to segment all time frames in a patient volume. We don't have reference annotations for all frames
        # but we still want this in cases where we would like to compute cardiac indices over the complete cardiac cycle
        # Loop over all time frames.
        cardiac_phase = "ALL"
        structures = {"ES": None, "ED": None}

        self._img4d, self._spacing = read_nifty(self._img4d_fname)
        self.original_spacing = self._spacing[:]
        self.number_of_frames = self._img4d.shape[0]
        for frame_id in range(self.number_of_frames):
            frame_id = int(frame_id)
            img = self._img4d[frame_id]
            if self._scale_intensities:
                img = rescale_intensities(img).astype(np.float32)
            # This is complete BULLSHIT! But we need to pass a valid object for this to the dataloader, it won't accept None
            # and we want our software to be generic as much as possible (same holds for variable "cardiac_phase")
            gt = img
            if self._resample:  # or self.original_spacing[-1] < 1.:
                raise NotImplementedError("ERROR - ACDCImage - _return_all_phases - resampling not implemented yet")
                # img, gt = self._do_resample(img, gt, self._spacing)
                # set new spacing
                # self._spacing = np.array([img.shape[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
            # NOTE: 'structures' is a list of integers [0=BG, 1=LV, 2=RV] indicating the labels the volume contains
            yield {'image': img.astype(np.float32), 'patient_id': self.patient_id, 'spacing': self._spacing,
                   'frame_id': frame_id, 'cardiac_phase': cardiac_phase, 'original_spacing': self.original_spacing,
                   'number_of_frames': self.number_of_frames}, \
                  {'labels': gt.astype(np.int16), 'patient_id': self.patient_id, 'spacing': self._spacing,
                   'frame_id': frame_id, 'cardiac_phase': cardiac_phase, 'original_spacing': self.original_spacing}

    @staticmethod
    def _check_spacing(spacing):
        # if not ndarray (e.g. tuple) convert to numpy. Otherwise dataloader gets stuck on tuples (expects np arrays)
        return spacing if isinstance(spacing, np.ndarray) else np.array(spacing).astype(np.float32)

    def get_frame_id(self, cardiac_phase):
        return int(self.info()[cardiac_phase])

    @property
    def patient_id(self):
        return self._patient_id

    def voxel_spacing(self):
        return np.array(self._img4d.header.get_zooms()[::-1]).astype(np.float32)

    def image4d(self):
        self._img4d = nib.load(self._img4d_fname)
        return self._img4d.get_data().T

    def shape(self):
        return self._img.header.get_data_shape()[::-1]

    def _do_resample(self, img, gt_lbl, sp):
        img = apply_2d_zoom_3d(img, sp, do_blur=True, new_spacing=self.new_spacing)
        gt_lbl = apply_2d_zoom_3d(gt_lbl, sp, order=0, do_blur=False, as_type=np.int, new_spacing=self.new_spacing)
        return img, gt_lbl

    @staticmethod
    def _determine_apex_base_slices(labels):
        """

        :param labels: numpy array of shape [z, y, x]
        :return: dict with 'A' = apex and 'B' = base keys. Values of dict are scalar slice ids
        """
        slice_ab = {'A': None, 'B': None}
        # Note: low-slice number => most basal slices / high-slice number => most apex slice
        # Note: assuming labels has one bg-class indicated as 0-label and shape [z, y, x]
        slice_ids = np.arange(labels.shape[0])
        # IMPORTANT: we sum over x, y and than check whether we'have a slice that has ZERO labels. So if
        # np.any() == True, this means there is a slice without labels.
        binary_mask = (np.sum(labels, axis=(1, 2)) == 0).astype(np.bool)
        if np.any(binary_mask):
            # we have slices (apex/base) that do not contain any labels. We assume that this can only happen
            # in the first or last slices e.g. [1, 1, 0, 0, 0, 0] so first 2 slice do not contain any labels
            slices_with_labels = slice_ids[binary_mask != 1]
            slice_ab['B'], slice_ab['A'] = int(min(slices_with_labels)), int(max(slices_with_labels))
        else:
            # all slices contain labels. We simply assume slice-idx=0 --> base and slice-idx = max#slice --> apex
            slice_ab['B'], slice_ab['A'] = int(min(slice_ids)), int(max(slice_ids))
        return slice_ab

    def info(self):
        try:
            self._info
        except AttributeError:
            self._info = dict()
            fname = self._path + '/Info.cfg'
            with open(fname, 'r') as f:
                for l in f:
                    k, v = l.split(':')
                    self._info[k.strip()] = v.strip()
        finally:
            return self._info


class ACDCDataset(Dataset):

    def __init__(self, dataset,
                 fold=0,
                 root_dir='/home/jorg/data/ACDC/all_cardiac_phases',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False,
                 resample_zaxis=False):

        self._root_dir = root_dir
        self.transform = transform
        self._resample = resample
        pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
        print("ACDCDataset - len({}) = {}".format(dataset, len(pat_nums)))
        images = list()
        references = list()
        spacing = list()
        org_spacing = list()
        ids = list()
        cardiac_phases = list()
        frame_ids = list()
        patient_ids = list()
        allidcs = np.empty((0, 2), dtype=int)
        avg_shape = np.zeros(3)

        for idx, patnum in tqdm(enumerate(pat_nums), desc='Load {} set fold {}'.format(dataset, fold)):
            img = ACDCImage(patnum, root_dir=root_dir, resample=self._resample, scale_intensities=rescale, resample_zaxis=resample_zaxis)
            ed, sp, ed_gt = img.ed()
            es, _,  es_gt = img.es()
            frame_id_ed = img.get_frame_id('ED')
            frame_id_es = img.get_frame_id('ES')

            images.append(ed)
            images.append(es)

            references.append(ed_gt.astype(int))
            references.append(es_gt.astype(int))

            spacing.append(sp)
            spacing.append(sp)
            org_spacing.append(img.original_spacing)
            org_spacing.append(img.original_spacing)

            cardiac_phases.append('ED')
            cardiac_phases.append('ES')
            frame_ids.append(frame_id_ed)
            frame_ids.append(frame_id_es)

            ids.append('{:03d} ED'.format(patnum))
            ids.append('{:03d} ES'.format(patnum))
            patient_ids.append('patient{:03d}'.format(patnum))
            patient_ids.append('patient{:03d}'.format(patnum))

            img_idx = (idx * 2)
            allidcs = np.vstack((allidcs, np.vstack((np.ones(len(ed)) * img_idx, np.arange(len(ed)))).T))
            img_idx = (idx * 2) + 1
            allidcs = np.vstack((allidcs, np.vstack((np.ones(len(es)) * img_idx, np.arange(len(es)))).T))

            avg_shape += es.shape
        avg_shape *= 1./(len(images) / 2)

        self._idcs = allidcs.astype(int)
        self._images = images
        self._references = references
        self._spacings = spacing
        self._org_spacings = org_spacing
        self._cardiac_phases = cardiac_phases
        self._frame_ids = frame_ids
        self._ids = ids
        self._patient_ids = patient_ids

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):

        img_idx, slice_idx = self._idcs[idx]

        sample = {'image': self._images[img_idx][slice_idx],
                  'reference': self._references[img_idx][slice_idx],
                  'spacing': self._spacings[img_idx],
                  'cardiac_phase': self._cardiac_phases[img_idx],
                  'frame_id': self._frame_ids[img_idx],
                  'patient_id': self._patient_ids[img_idx],
                  'original_spacing': self._org_spacings[img_idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

