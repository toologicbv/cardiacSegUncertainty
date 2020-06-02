
import nibabel as nib
import numpy as np
import os
import glob
import copy

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets.ACDC.v1.utilities import split_acdc_dataset, apply_2d_zoom_3d
from datasets.ACDC.v1.acdc_transformers import get_dataset_settings

from datasets.ACDC.v1.settings import acdc_settings3d


def resample2d(data_item, debug_info=None):
    # spacing has shape [#slices, IH, IW]. We resample to 1.4mm x 1.4mm for the last 2 dimensions
    spacing = data_item['spacing']
    if len(spacing) == 3:
        new_spacing = (tuple((spacing[0],)) + acdc_settings3d.voxel_spacing)
    elif len(spacing) == 4:
        new_spacing = (tuple((spacing[1],)) + acdc_settings3d.voxel_spacing)
    else:
        raise ValueError("ERROR - resample2d - spacing not supported ", spacing)

    for item in ['image_ed', 'image_es']:
        data_item[item] = apply_2d_zoom_3d(data_item[item], spacing, do_blur=True,
                                                        new_spacing=acdc_settings3d.voxel_spacing)

    for item in ['label_ed', 'label_es']:
        mycopy = copy.deepcopy(data_item[item])
        data_item[item] = apply_2d_zoom_3d(data_item[item], spacing, order=0, do_blur=False,
                                                        as_type=np.int, new_spacing=acdc_settings3d.voxel_spacing)
        for z in range(mycopy.shape[0]):
            if not np.all(np.unique(data_item[item][z, :, :]) == np.unique(mycopy[z, :, :])):
                print("WARNING - slice {} - unique labels not anymore the same! ".format(z) + debug_info)
    data_item['original_spacing'] = spacing
    data_item['spacing'] = new_spacing
    return data_item


def acdc_fold_patient(fold_id, dataset_type, src_data_path, resample=False, limited_load=False, patient_id=None,
                      verbose=False):
    assert not (fold_id is None and patient_id is None)
    if fold_id is not None:
        train_pat_nums, val_pat_nums = split_acdc_dataset(fold_id)
    else:
        train_pat_nums, val_pat_nums = [], []
    if dataset_type == 'train':
        if patient_id is None:
            patient_ids = train_pat_nums
        else:
            patient_ids = [patient_id]
    elif dataset_type == 'test':
        if patient_id is None:
            # val_pat_nums = np.concatenate((val_pat_nums, train_pat_nums[::3]), axis=0)
            # print("WARNING - val_pat_nums extended to {}".format(len(val_pat_nums)))
            patient_ids = val_pat_nums
        else:
            patient_ids = [patient_id]
    else:
        raise ValueError("{} not supported".format(dataset_type))
    if verbose:
        print(">>>>>>>>>>>> INFO - ACDC Dataset - Loading from {}".format(src_data_path))
    data = {}
    if limited_load:
        patient_ids = patient_ids[:5]

    if patient_id is None:
        mygenerator = tqdm(patient_ids, desc='Loading {} dataset'.format(dataset_type))
    else:
        mygenerator = patient_ids
    for patid in mygenerator:
        acdc_edes_image = ACDCImageEDES(patid, root_dir=acdc_settings3d.data_path)
        data_item = acdc_edes_image.get_item()
        # previous version
        # data_item = get_acdc_data_item(patid, fnames_images, fnames_labels)
        if resample:
            data_item = resample2d(data_item,  debug_info=str(patid))
            # save_resampled(data_item, fnames_images, fnames_labels) disabled 03-03-2019

        data[patid] = data_item
        del data_item

    return data


class ACDCImageEDES(object):
    def __init__(self, number, root_dir=acdc_settings3d.data_path, scale_intensities=True):
        """
        IMPORTANT: After loading: numpy array gets reshaped to [z, y, x] z=#slices

        :param number: patient id (1...100)
        :param root_dir: data root dir
        :param scale_intensities: boolean
        """

        self._number = number
        self.patient_id = "patient{:03d}".format(number)
        self._path = os.path.join(root_dir, 'patient{:03d}'.format(number))
        self.info()
        frame_id_ed, frame_id_es = int(self.info()['ED']), int(self.info()['ES'])
        self._img_fname_ed = os.path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number,
                                                                                                frame_id_ed))
        self._check_file_exists(self._img_fname_ed)
        self._img_fname_es = os.path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number,
                                                                                                frame_id_es))
        self._check_file_exists(self._img_fname_es)
        self._lbl_fname_ed = self._img_fname_ed.replace(".nii.gz",
                                                        "_gt.nii.gz")
        self._check_file_exists(self._lbl_fname_ed)
        self._lbl_fname_es = self._img_fname_es.replace(".nii.gz",
                                                        "_gt.nii.gz")
        self._check_file_exists(self._lbl_fname_es)
        self._image = {'image_ed': None, 'image_es': None}
        self._scale_intensities = scale_intensities
        self.frame_id_ed = frame_id_ed
        self.frame_id_es = frame_id_es

    @staticmethod
    def _check_file_exists(filename):
        if not os.path.isfile(filename):
            raise FileExistsError("ERROR - ARVCEDESImage - file does not exist {}".format(filename))

    def voxel_spacing(self):
        # important we reverse voxel spacing AND shape because after loading data we reshape to [z, y, x]
        # whereas NIFTI images/labels are stored in [x, y, z]
        return self._image['image_ed'].header.get_zooms()[::-1]

    def shape(self):
        return self._image['image_ed'].header.get_data_shape()[::-1]

    def data(self):
        try:
            self._img_data, self._lbl_data
        except AttributeError:
            self._img_data = {'image_ed': None, 'image_es': None}
            self._image = {'image_ed': nib.load(self._img_fname_ed), 'image_es': nib.load(self._img_fname_es)}
            self._lbl_data = {'label_ed': nib.load(self._lbl_fname_ed), 'label_es': nib.load(self._lbl_fname_es)}
            for ikey in self._image.keys():
                # IMPORTANT: numpy array gets reshaped to [z, y, x] z=#slices
                data = self._image[ikey].get_data(caching='fill').transpose(2, 1, 0)
                if self._scale_intensities:
                    data = self._rescale_intensities_per_slice(data)
                    # data = self._rescale_intensities_jelmer(data)
                    # print("WARNING - Normalizing!")
                    # data = self._normalize(data)

                self._img_data[ikey] = data
            for ikey in self._lbl_data.keys():
                # also need to reshape labels to [z, y, x]
                self._lbl_data[ikey] = self._lbl_data[ikey].get_data(caching='fill').transpose(2, 1, 0)

        finally:
            return self._img_data, self._lbl_data

    def get_item(self):
        _ = self.data()
        es_apex_base_slices = self._determine_apex_base_slices(self._lbl_data['label_es'])
        ed_apex_base_slices = self._determine_apex_base_slices(self._lbl_data['label_ed'])
        return {'image_ed': self._img_data['image_ed'], 'label_ed': self._lbl_data['label_ed'],
                'image_es': self._img_data['image_es'], 'label_es': self._lbl_data['label_es'],
                'spacing': self.voxel_spacing(), 'origin': None, 'frame_id_ed': self.frame_id_ed,
                'patient_id': self.patient_id, 'patid': self._number, 'num_of_slices': self.shape()[0],
                'frame_id_es': self.frame_id_es, 'type_extra_input': None, 'info': self._info,
                'apex_base_es': es_apex_base_slices, 'apex_base_ed': ed_apex_base_slices}

    @staticmethod
    def _rescale_intensities(img_data, percentile=acdc_settings3d.int_percentiles):
        min_val, max_val = np.percentile(img_data, percentile)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _rescale_intensities_per_slice(img_data, percentile=acdc_settings3d.int_percentiles):
        min_val, max_val = np.percentile(img_data, percentile, axis=(1, 2), keepdims=True)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _rescale_intensities_jelmer(img_data, percentile=acdc_settings3d.int_percentiles):
        min_val, max_val = np.percentile(img_data, percentile)
        data = (img_data.astype(float) - min_val) / (max_val - min_val)
        data -= 0.5
        min_val, max_val = np.percentile(data, percentile)
        return ((data - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _determine_apex_base_slices(labels):
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

    @staticmethod
    def _normalize(img_data):
        return (img_data - np.mean(img_data)) / np.std(img_data)

    def info(self):
        try:
            self._info
        except AttributeError:
            self._info = dict()
            fname = os.path.join(self._path, 'Info.cfg')
            with open(fname, 'rU') as f:
                for l in f:
                    k, v = l.split(':')
                    self._info[k.strip()] = v.strip()
        finally:
            return self._info


class ACDC2017DataSet(Dataset):

    train_path = "train"
    val_path = "validate"
    image_path = "images_resize"
    label_path = "reference_resize"

    pixel_dta_type = 'float32'
    new_voxel_spacing = 1.4

    """
        IMPORTANT: For each patient we load four files (1) ED-image (2) ED-reference (3) ES-image (4) ES-reference
                   !!! ED image ALWAYS ends with 01 index
                   !!! ES image has higher index

                   HENCE ES IMAGE IS THE SECOND WE'RE LOADING in the _load_file_list method
    """

    def __init__(self, fold_id=0, src_data_path=acdc_settings3d.data_path , dataset_type="train", mode=None,
                 type_extra_input=None, transform=None, **kwargs):

        assert dataset_type in ["train", "test"]

        self.fold_id = fold_id
        self.resample = kwargs['resample']
        self.transform = transform
        self.limited_load = kwargs['limited_load']
        self.dataset_type = dataset_type
        self.dataset_settings = get_dataset_settings(mode)
        self.dataset_settings["mode"] = mode
        self.data_formatter = self.dataset_settings["train_data_formatter"] if dataset_type == "train" else \
            self.dataset_settings["test_data_formatter"]
        self.type_extra_input = type_extra_input
        self.num_of_classes = self.dataset_settings["num_classes"]
        self.num_of_cardiac_phases = int(self.num_of_classes / 4)
        self.images, self.labels, self.extra_labels = [], [], {}
        self.trans_dict = {}

        self.extra_dta_handler = None
        self.num_of_studies = 0
        self.length_dataset = 0
        self.src_data_path = src_data_path

        patient_id = None if 'patient_id' not in kwargs.keys() else kwargs['patient_id']
        self._data = acdc_fold_patient(fold_id, dataset_type, self.src_data_path, patient_id=patient_id,
                                       resample=self.resample, limited_load=self.limited_load)

        self.num_of_studies = len(self._data)
        self.create()
        self.num_input_channels = self.dataset_settings["num_input_channels"]
        # print("INFO - Dataset - Num of channels {}".format(self.num_input_channels))

    def create(self):
        self.images, self.labels, self.extra_labels = [], [], {}
        if len(self._data) == 1:
            mygenerator = self._data.items()
        else:
            mygenerator = tqdm(self._data.items(), desc="Using {}".format(self.data_formatter.__class__.__name__))
        for patid, data_item in mygenerator:
            if self.dataset_type == "train":
                self._add_data_as_slices(data_item)
            else:
                self._add_data(data_item)

    def _add_data(self, data_item):
        data_item = self.data_formatter(data_item)
        # IMPORTANT: data_item can contain more than ONE tuple. In case we choose dataset mode = "separate"
        # the data_item contains 2 (!) tuples: (1) for ES and (2) for ED. And we add them separately to the dataset
        # in this LOOP. Furthermore, data_tuple is a 2-tuple. (1) contains image (2) labels. Both are dictionaries.
        pat_dataset_idx = []
        for data_tuple in data_item:
            patid = data_tuple[0]['patid']
            pat_dataset_idx.append(self.length_dataset)
            self.images.append(data_tuple[0]), self.labels.append(data_tuple[1])
            self.length_dataset += 1
        self.trans_dict[patid] = pat_dataset_idx

    def _add_data_as_slices(self, data_item):

        # add concatenation of ED+ES (dim0) for images and labels
        pat_dataset_ids = []
        self.trans_dict[data_item['patid']] = []

        for z in range(data_item['num_of_slices']):
            data_item_slice = self.data_formatter(data_item, z)
            pat_dataset_ids = self._append(data_item_slice, pat_dataset_ids, z)
            self.length_dataset += 1
            self.trans_dict[data_item['patid']].append(self.length_dataset)

        self.extra_labels[data_item['patid']] = pat_dataset_ids

    def _append(self, data_item_slice, pat_dataset_ids, slice_id):
        """

        :param data_item_slice: (1) can be a tuple consisting of 2 dictionaries e.g. each dict containing
                                    keys: 'image', 'label', 'cardiac_phase' (ES/ED or ESED), 'spacing'
                                (2) can be a tuple of a tuple, in case we split the ES and ED slices
                                    then the first tuple (of tuples) contains the ES image/labels
                                    and the second tuple contains the ED image/labels (used for SeparateCardiacPhase obj)
        :return:
        """

        for data_tuple in data_item_slice:
            # data_tuple should be a tuple consisting of 2 dictionaries: 1: image, 2: label
            self.images.append(data_tuple[0]), self.labels.append(data_tuple[1])
            self.length_dataset += 1
            pat_dataset_ids.append(tuple((len(self) - 1, slice_id)))

        return pat_dataset_ids

    def __getitem__(self, index):
        assert index <= self.__len__()
        data_tuple = copy.deepcopy(tuple((self.images[index], self.labels[index])))
        if self.transform:
            data_tuple = self.transform(data_tuple)
        return data_tuple

    def __len__(self):
        return len(self.images)

    def get_mode(self):
        return self.dataset_settings["mode"]

    def info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Create dataset')
    # args = parser.parse_args()
    from torch.utils.data.sampler import RandomSampler
    import utils.acdc.acdc_transformers as dcnn_trans
    from datasets.ACDC.v1.settings import acdc_settings3d

    rs = np.random.RandomState(78346)
    fold_id = 0
    num_classes = 4
    extra_input = 'LowQualityDetection'
    dataset_train = ACDC2017DataSet(fold_id=fold_id, dataset_type="train", resample=True, limited_load=True,
                                    src_data_path=acdc_settings3d.data_path,
                                   type_extra_input=extra_input, mode="separate",
                                   transform=transforms.Compose([dcnn_trans.DCNNPadding(acdc_settings3d.pad_size,
                                                                                        is_volume=False),
                                                                 dcnn_trans.RandomRotation(rs=rs, only_inplane=True),
                                                                 dcnn_trans.RandomMirroring(rs=rs),
                                                                 dcnn_trans.RandomCrop(
                                                                     acdc_settings3d.batch_batch_shape_wpadding,
                                                                     acdc_settings3d.batch_batch_shape, rs=rs),
                                                                 dcnn_trans.LabelBinarizer(num_classes=num_classes,
                                                                                           keep_multi_labels=True),
                                                                 dcnn_trans.ToTensor(is_volume=False)]))

    # test_set = get_acdc_test_set(fold_id=0, mode_dataset="separate",
    #                                 type_extra_input=None, limited_load=False, resample=False)
    tra_sampler = RandomSampler(dataset_train, replacement=True, num_samples=16)
    data_loader_training = DataLoader(dataset_train, batch_size=8, sampler=tra_sampler)
    for it, test_batch in tqdm(enumerate(data_loader_training), desc="train"):
        data_image, data_label = test_batch
        # if data_image['patid'] == 16:
        print(data_image['image'].shape, data_label['label'].shape, data_label['extra_data'].shape)
    #     print("\n", it, data_image['patid'], data_label['label'].shape, data_image['phase_id'])
    #
    # print(len(dtaset.images), len(dtaset.labels))
    # del dtaset
