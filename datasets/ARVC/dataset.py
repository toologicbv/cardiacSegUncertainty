import copy
import numpy as np
import os
import glob
import yaml
import shutil

from tqdm import tqdm
from torch.utils.data import Dataset
from datasets.data_config import get_config
from datasets.common import read_nifty, apply_2d_zoom_3d, get_arvc_datasets

arvc_data_settings = get_config('ARVC')


def do_limit_load(p_dataset_filelist):
    return p_dataset_filelist[:arvc_data_settings.limited_load_max]


def load_data(data_config, limited_load=False, search_mask="*.nii.gz", **kwargs):
    '''

    :return:
    '''
    load_ref_labels = False if "load_ref_labels" not in kwargs.keys() else kwargs['load_ref_labels']
    resample = kwargs.get('resample', False)
    search_mask_img = os.path.expanduser(os.path.join(data_config.short_axis_dir, search_mask))
    files_to_load = glob.glob(search_mask_img)
    files_to_load.sort()
    if len(files_to_load) == 0:
        raise ValueError("ERROR - ARVC data - Can't find any files to load in {}".format(search_mask))

    if limited_load:
        files_to_load = do_limit_load(files_to_load)

    images = {}
    for file_name in tqdm(files_to_load, desc='Load {}'.format(data_config.dataset)):
        if load_ref_labels:
            filename_ref_labels = file_name.replace(data_config.short_axis_dir, data_config.ref_label_dir)
        else:
            filename_ref_labels = None
        img = ARVCImage(file_name, filename_ref_labels=filename_ref_labels, resample=resample)
        images[img.patient_id] = []
        if load_ref_labels and img.has_labels:
            for sample_img, sample_lbl in img.ed():
                images[img.patient_id].append(sample_img)
            for sample_img, sample_lbl in img.es():
                images[img.patient_id].append(sample_img)
        print("INFO: {} {}".format(img.patient_id, len(images[img.patient_id])))
    return images


def arvc_get_evaluate_set(dataset, limited_load=False, resample=False, rescale=True, patid=None, all_frames=False):
    """
    We use this function during validation and testing. Different than for the ARVCDataSet object which returns
    slices that are used for training and hence transformed e.g. rotation, mirroring etc.

    :param dataset:
    :param limited_load:
    :param resample:
    :param rescale:
    :param patid: one patient ID to process (string)
    :param all_frames: boolean, if TRUE, we get all time frames of a patient ignoring reference labels
    :return:
    """
    dta_settings = get_config('ARVC')
    assert dataset in dta_settings.datasets
    files_to_load = get_arvc_datasets()[dataset]
    if patid is not None:
        files_to_load = [fname for fname in files_to_load if patid in fname]
        if len(files_to_load) == 0:

            raise ValueError("ERROR - {} is not a valid patient id".format(patid))
    if limited_load:
        files_to_load = do_limit_load(files_to_load)
    patient_data_idx = {}

    idx = 0

    for filename in tqdm(files_to_load, desc="Load {} set".format(dataset)):
        filename_ref_labels = filename.replace(dta_settings.short_axis_dir, dta_settings.ref_label_dir)
        img = ARVCImage(filename, filename_ref_labels=filename_ref_labels, rescale=rescale,
                        resample=resample)
        if all_frames:

            patient_data_idx[img.patient_id] = []
            for sample_img, sample_lbl in img.all():
                sample = {'image': sample_img['image'],
                          'reference': sample_lbl['labels'],
                          'spacing': sample_img['spacing'],
                          'direction': sample_img['direction'],
                          'origin': sample_img['origin'],
                          'frame_id': sample_img['frame_id'],
                          'cardiac_phase': sample_img['cardiac_phase'],
                          'structures': sample_lbl['structures'],
                          'ignore_label': sample_lbl['ignore_label'],
                          'original_spacing': sample_lbl['original_spacing'],
                          'patient_id': sample_lbl['patient_id'],
                          'number_of_frames': sample_img['number_of_frames']}
                patient_data_idx[img.patient_id].append(idx)
                yield sample
                idx += 1

        if img.has_labels and not all_frames:
            patient_data_idx[img.patient_id] = []
            for sample_img, sample_lbl in img.ed():
                sample = {'image': sample_img['image'],
                          'reference': sample_lbl['labels'],
                          'spacing': sample_img['spacing'],
                          'direction': sample_img['direction'],
                          'origin': sample_img['origin'],
                          'frame_id': sample_img['frame_id'],
                          'cardiac_phase': sample_img['cardiac_phase'],
                          'structures': sample_lbl['structures'],
                          'ignore_label': sample_lbl['ignore_label'],
                          'original_spacing': sample_lbl['original_spacing'],
                          'patient_id': sample_lbl['patient_id'],
                          'number_of_frames': sample_img['number_of_frames']}
                patient_data_idx[img.patient_id].append(idx)
                yield sample
                idx += 1

            for sample_img, sample_lbl in img.es():
                sample = {'image': sample_img['image'],
                          'reference': sample_lbl['labels'],
                          'spacing': sample_img['spacing'],
                          'direction': sample_img['direction'],
                          'origin': sample_img['origin'],
                          'frame_id': sample_img['frame_id'],
                          'cardiac_phase': sample_img['cardiac_phase'],
                          'structures': sample_lbl['structures'],
                          'ignore_label': sample_lbl['ignore_label'],
                          'original_spacing': sample_lbl['original_spacing'],
                          'patient_id': sample_lbl['patient_id'],
                          'number_of_frames': sample_img['number_of_frames']}
                patient_data_idx[img.patient_id].append(idx)
                yield sample
                idx += 1


# [234 122 143 219  73 238 209 198]
# [232 113 157   5  68  18 206  28]

class ARVCDataset(Dataset):

    """

    """

    def __init__(self, dataset,
                 root_dir='~/data/ARVC/',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False):

        self._root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self._resample = resample
        self._rescale = rescale
        self.dta_settings = get_config('ARVC')
        assert dataset in self.dta_settings.datasets
        files_to_load = get_arvc_datasets()[dataset]
        if limited_load:
            files_to_load = do_limit_load(files_to_load)
        images = list()
        references = list()
        ids = list()
        patient_data_idx = {}
        allidcs = np.empty((0, 2), dtype=int)

        idx = 0
        for filename in tqdm(files_to_load, desc="Load {} set".format(dataset)):

            filename_ref_labels = filename.replace(self.dta_settings.short_axis_dir, self.dta_settings.ref_label_dir)
            img = ARVCImage(filename, filename_ref_labels=filename_ref_labels, rescale=self._rescale,
                            resample=self._resample)
            if img.has_labels:
                patient_data_idx[img.patient_id] = []
                for sample_img, sample_lbl in img.ed():
                    images.append(sample_img)
                    references.append(sample_lbl)
                    ids.append(idx)

                    patient_data_idx[img.patient_id].append(idx)
                    num_slices = len(sample_lbl['labels'])
                    allidcs = np.vstack((allidcs, np.vstack((np.ones(num_slices) * idx, np.arange(num_slices))).T))
                    idx += 1
                for sample_img, sample_lbl in img.es():
                    images.append(sample_img)
                    references.append(sample_lbl)
                    ids.append(idx)
                    patient_data_idx[img.patient_id].append(idx)
                    num_slices = len(sample_lbl['labels'])
                    allidcs = np.vstack((allidcs, np.vstack((np.ones(num_slices) * idx, np.arange(num_slices))).T))
                    idx += 1
        self._idcs = allidcs.astype(int)
        self._images = images
        self._references = references
        self._ids = ids
        self._patient_data_idx = patient_data_idx

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):

        img_idx, slice_idx = self._idcs[idx]

        sample = {'image': self._images[img_idx]['image'][slice_idx],
                  'reference': self._references[img_idx]['labels'][slice_idx],
                  'spacing': self._images[img_idx]['spacing'],
                  'original_spacing': self._images[img_idx]['spacing'],
                  'patient_id': self._images[img_idx]['patient_id'],
                  'direction': self._images[img_idx]['direction'],
                  'origin': self._images[img_idx]['origin'],
                  'frame_id': self._images[img_idx]['frame_id'],
                  'cardiac_phase': self._images[img_idx]['cardiac_phase'],
                  'structures': self._references[img_idx]['structures'],
                  'ignore_label': self._references[img_idx]['ignore_label'],
                  'number_of_frames': self._images[img_idx]['number_of_frames']}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ARVCImage(object):

    new_spacing = np.array([1, 1.4, 1.4])
    n_classes = 3

    def __init__(self, filename, filename_ref_labels=None, rescale=True, resample=False):
        self._filename = filename
        self._rescale = rescale
        self._resample = resample
        # read_nifty returns numpy array of shape [#frames, z, y, x]
        self._img, self._spacing, self._direction, self._origin = read_nifty(self._filename, get_extra_info=True)
        self._spacing = self._spacing[1:]  # skip frame dimension for spacing

        self.original_spacing = self._spacing
        self.number_of_frames, self.number_of_slices, _, _ = self._img.shape
        self._filename_ref_labels = filename_ref_labels
        self.has_labels = True
        self._labels = None
        self._info = None

        if self._filename_ref_labels is not None:
            if os.path.isfile(self._filename_ref_labels):
                self._labels, _ = read_nifty(self._filename_ref_labels)
            else:
                self.has_labels = False
            # yaml file that contains dictionary with info about ES ED phase numbers and tissue structure labels
            self._filename_info = filename_ref_labels.replace(".nii.gz", ".yml")
            if os.path.isfile(self._filename_info):
                with open(self._filename_info, 'r') as fp:
                    self._info = yaml.load(fp, Loader=yaml.FullLoader)

        stem_filename = os.path.splitext(os.path.basename(self._filename))[0]
        self.patient_id = stem_filename.strip('.nii.gz')

    def ed(self):
        # remember ARVC can have multiple phase IDs for ED/ES
        return self._return_phase_images('ED')

    def es(self):
        return self._return_phase_images('ES')

    def all(self):
        return self._return_all_phases()

    def _return_all_phases(self):
        # Use this method to segment all time frames in a patient volume. We don't have reference annotations for all frames
        # but we still want this in cases where we would like to compute cardiac indices over the complete cardiac cycle
        # Loop over all time frames.
        cardiac_phase = "ALL"
        structures = {"ES": None, "ED": None}
        for frame_id in range(self.number_of_frames):
            frame_id = int(frame_id)
            # ignore all labels, set them to 1
            ignore_label = np.zeros(self.n_classes)
            img = self._img[frame_id]
            if self._rescale:
                img = self._rescale_intensities(img)
            # This is complete BULLSHIT! But we need to pass a valid object for this to the dataloader, it won't accept None
            # and we want our software to be generic as much as possible (same holds for variable "cardiac_phase")
            gt = img
            if self._resample:
                img, gt = self._do_resample(img, gt, self._spacing)
                # set new spacing
                self._spacing = np.array([img.shape[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
            # NOTE: 'structures' is a list of integers [0=BG, 1=LV, 2=RV] indicating the labels the volume contains
            yield {'image': img.astype(np.float32), 'patient_id': self.patient_id, 'spacing': self._spacing,
                   'frame_id': frame_id, 'cardiac_phase': cardiac_phase, 'original_spacing': self.original_spacing,
                   'number_of_frames': self.number_of_frames, 'direction': self._direction, 'origin': self._origin}, \
                  {'labels': gt.astype(np.int16), 'patient_id': self.patient_id, 'spacing': self._spacing,
                   'frame_id': frame_id, 'cardiac_phase': cardiac_phase, 'original_spacing': self.original_spacing,
                   'structures': structures, 'ignore_label': ignore_label}

    def _return_phase_images(self, cardiac_phase):
        # remember ARVC can have multiple phase IDs for ED/ES
        for frame_id, structures in self._info[cardiac_phase].items():
            frame_id = int(frame_id)
            ignore_label = np.zeros(self.n_classes)
            if len(structures) == 1:
                # volume does not contain LV and RV
                # set indices to 1 of the labels that we should ignore, hence that are not in list structures
                ignore_label[np.array([i for i in np.arange(1, self.n_classes) if i not in structures])] = 1
            # pick the specific frame id, hence img has [z, y, x]
            img = self._img[frame_id]
            if self._rescale:
                img = self._rescale_intensities(img)
            gt = self._labels[frame_id]
            if self._resample:
                img, gt = self._do_resample(img, gt, self._spacing)
                # set new spacing
                self._spacing = np.array([img.shape[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
            # NOTE: 'structures' is a list of integers [0=BG, 1=LV, 2=RV] indicating the labels the volume contains
            yield {'image': img.astype(np.float32), 'patient_id': self.patient_id, 'spacing': self._spacing,
                   'frame_id': frame_id, 'cardiac_phase': cardiac_phase, 'original_spacing': self.original_spacing,
                   'number_of_frames': self.number_of_frames, 'direction': self._direction, 'origin': self._origin}, \
                  {'labels': gt.astype(np.int16), 'patient_id': self.patient_id, 'spacing': self._spacing,
                   'frame_id': frame_id, 'cardiac_phase': cardiac_phase, 'original_spacing': self.original_spacing,
                   'structures': structures, 'ignore_label': ignore_label}

    def _do_resample(self, img, gt_lbl, sp):
        img = apply_2d_zoom_3d(img, sp, do_blur=True, new_spacing=self.new_spacing)
        gt_lbl = apply_2d_zoom_3d(gt_lbl, sp, order=0, do_blur=False, as_type=np.int, new_spacing=self.new_spacing)
        return img, gt_lbl

    @staticmethod
    def _rescale_intensities(img_data, percentile=(1, 99)):
        min_val, max_val = np.percentile(img_data, percentile)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _rescale_intensities_per_slice(img_data, percentile=(1, 99)):
        min_val, max_val = np.percentile(img_data, percentile, axis=(1, 2), keepdims=True)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)


if __name__ == '__main__':
    data_config = get_config('ARVC')
    input_dir = "~/data/ARVC"
    # images = load_data(data_config, limited_load=True, search_mask="*.nii.gz", load_ref_labels=True, resample=False)
    # print(len(images))
    # dataset = ARVCDataset(root_dir=input_dir, dataset="training", limited_load=True, rescale=True, resample=False)
    # sample = dataset[10]
    # print(sample.keys())
    # print(sample['image'].shape, sample['reference'].shape, sample['structures'])
    data = arvc_get_evaluate_set("training", limited_load=True, resample=False, rescale=True)
    for dta_item in data:
        print(dta_item['image'].shape, dta_item['frame_id'], dta_item['cardiac_phase'], dta_item['structures'],
              dta_item['ignore_label'])
