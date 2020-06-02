import os
import numpy as np


class Config(object):

    def __init__(self):
        self.dataset = None
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/'))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'images')
        self.ref_label_dir = os.path.join(self.data_root_dir, 'ref_tissue_labels')
        self.tissue_structure_labels = None
        self.split_file = None
        self.datasets = ['training', 'validation', 'test']
        self.limited_load_max = 5
        self.dt_margins = None


class ConfigARVC(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'images')
        self.ref_label_dir = os.path.join(self.data_root_dir, 'ref_tissue_labels')
        self.tissue_structure_labels = {0: 'BG', 1: 'LV', 2: 'RV'}
        self.split_file = os.path.join(self.data_root_dir, "train_test_split.yaml")
        self.datasets = ['training', 'validation', 'test']
        self.limited_load_max = 10
        # transfer learning. Assuming we train on ACDC and test on ARVC. We need to translate ARVC classes
        # to ACDC classes
        self.cls_translate = {1: 3, 2: 1}


class ConfigACDC(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'all_cardiac_phases')
        self.ref_label_dir = None
        self.tissue_structure_labels = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1.4, 1.4]).astype(np.float32)
        self.dt_margins = (4.6, 3.1)


class TransferDataset(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'images')
        self.ref_label_dir = None
        self.tissue_structure_labels = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1.4, 1.4]).astype(np.float32)
        self.dt_margins = (4.6, 3.1)


def get_config(dataset="ARVC"):
    if dataset == "ARVC":
        return ConfigARVC(dataset=dataset)
    elif dataset == "ACDC" or dataset == "ACDC_full":
        if dataset == "ACDC_full":
            dataset = 'ACDC'
        return ConfigACDC(dataset=dataset)
    elif dataset == "vumc_pulmonary":
        return TransferDataset(dataset=dataset)



