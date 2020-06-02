import numpy as np
import torch

from utils.common import one_hot_encoding


def get_dataset_settings(mode_dataset):

    pad_size = 65
    if mode_dataset[:6].lower() == "single":
        phase_id = mode_dataset[6:]
        mode_dataset = mode_dataset[:6].lower()
    else:
        phase_id = "default"

    dataset_settings = {'combined': {"train_data_formatter": ESEDCombinedSlice(), "num_classes": 8,
                                     "test_data_formatter": ESEDCombinedVolume(), "phase_id": "ESED",
                                     "num_input_channels": 2},
                        'single': {"train_data_formatter": SinglePhaseSlice(phase_id),
                                   "num_classes": 4, 'phase_id': phase_id, "num_input_channels": 1,
                                   "test_data_formatter": SinglePhaseVolume(phase_id)},
                        'separate': {"train_data_formatter": SeparatePhaseSlice(), "num_classes": 4,
                                     "test_data_formatter": SeparatePhaseVolume(), "phase_id": "ESED",
                                     "num_input_channels": 1},
                        'full_cardiac_cycle': {"num_input_channels": 1, "num_classes": 4,
                                               "test_data_formatter": CompleteCardiacCycleVolume(pad_size=pad_size)}
                }
    return dataset_settings[mode_dataset]


def concatenate_slice_with_extra_input(image, extra_channels):

    # we're supporting arrays with #dims 2 or 3
    if extra_channels is not None:
        if extra_channels.ndim == 2:
            image = np.concatenate((image, extra_channels[np.newaxis]), axis=0)
        elif extra_channels.ndim == 3:
            #
            image = np.concatenate((image, extra_channels), axis=0)
    return image


def concatenate_vol_with_extra_input(image, extra_channels):
    """

    :param image: has shape [C, w, h, d] or [C, w, h, d]
    :param extra_channels: has shape [w, h, d] or [C, w, h, d]
    :return:
    """
    # we're supporting arrays with #dims 3 or 4
    if extra_channels is not None:
        if extra_channels.ndim == 3:
            image = np.concatenate((image, extra_channels[np.newaxis]), axis=0)
        elif extra_channels.ndim == 4:
            #
            image = np.concatenate((image, extra_channels), axis=0)
    return image


class ESEDCombinedSlice(object):

    def __init__(self):
        self.evaluation_per_phase = False
        self.has_extra_data = False

    def __call__(self, *args):
        data_item, idx = args
        if 'extra_data' in list(data_item.keys()):
            self.has_extra_data = True
        else:
            self.has_extra_data = False
        return self._concatenate(data_item, idx)

    def _concatenate(self, *args):
        data_item, idx = args
        images = np.concatenate((data_item['image_es'][np.newaxis, idx],
                                 data_item['image_ed'][np.newaxis, idx]))
        labels = np.concatenate((data_item['label_es'][np.newaxis, idx],
                                 data_item['label_ed'][np.newaxis, idx]))

        if self.has_extra_data:
            images = concatenate_slice_with_extra_input(images, data_item['extra_data'][idx])
        return [tuple(({'image': images, 'phase_id': "ESED", 'spacing': data_item['spacing'],
                        'original_spacing': data_item['original_spacing'],
                        'patid': data_item['patid'], "num_of_slices": data_item['num_of_slices']},
                      {'label': labels, 'phase_id': "ESED", 'spacing': data_item['spacing'],
                       'patid': data_item['patid'], "num_of_slices": data_item['num_of_slices']}))]


class ESEDCombinedVolume(ESEDCombinedSlice):

    def __call__(self, *args):
        data_item = args[0]
        self.has_extra_data = False
        return self._concatenate(data_item)

    def _concatenate(self, data_item):
        if 'extra_data' in list(data_item.keys()):
            self.has_extra_data = True
        else:
            self.has_extra_data = False
        images = np.concatenate((data_item['image_es'][np.newaxis],
                                 data_item['image_ed'][np.newaxis]))
        labels = np.concatenate((data_item['label_es'][np.newaxis],
                                 data_item['label_ed'][np.newaxis]))
        if self.has_extra_data:
            images = concatenate_slice_with_extra_input(images, data_item['extra_data'])
        return [tuple(({'image': images, 'phase_id': "ESED", 'spacing': data_item['spacing'],
                        'original_spacing': data_item['original_spacing'],
                        'patid': data_item['patid'], "num_of_slices": data_item['num_of_slices']},
                      {'label': labels, 'phase_id': "ESED", 'spacing': data_item['spacing'],
                       'patid': data_item['patid'], "num_of_slices": data_item['num_of_slices']}))]


class SinglePhaseSlice(object):

    def __init__(self, cardiac_phase=None):
        assert cardiac_phase in ["ES", "ED", "default"]
        self.evaluation_per_phase = False
        self.cardiac_phase = cardiac_phase
        self.image_tag = 'image_' + cardiac_phase.lower()
        self.label_tag = 'label_' + cardiac_phase.lower()
        self.phase_id_tag = 'frame_id_' + cardiac_phase.lower()
        self.has_extra_data = False

    def __call__(self, *args):
        data_item = args[0]
        idx = args[1]
        if 'extra_data' in list(data_item.keys()):
            self.has_extra_data = True
        else:
            self.has_extra_data = False

        image_item = {'image': data_item[self.image_tag][idx], 'phase_id': self.cardiac_phase,
                      'spacing': data_item['spacing'], 'patid': data_item['patid'],
                      'original_spacing': data_item['original_spacing'],
                      'frame_id': data_item[self.phase_id_tag],
                      "slice_id": idx, "num_of_slices": data_item['num_of_slices']}

        if self.has_extra_data:
            image_item['extra_data'] = data_item['extra_data'][idx]
        return [tuple((image_item,
                      {'label': data_item[self.label_tag][idx], 'patid': data_item['patid'],
                       'spacing': data_item['spacing'], "slice_id": idx, "num_of_slices": data_item['num_of_slices']}))]


class SinglePhaseVolume(SinglePhaseSlice):

    def __call__(self, *args):
        data_item = args[0]
        if 'extra_data' in list(data_item.keys()):
            self.has_extra_data = True
        else:
            self.has_extra_data = False

        image_item = {'image': data_item[self.image_tag], 'phase_id': self.cardiac_phase, 'patid': data_item['patid'],
                      'spacing': data_item['spacing'], "num_of_slices": data_item['num_of_slices'],
                      'original_spacing': data_item['original_spacing'],
                      'frame_id': data_item[self.phase_id_tag]}

        if self.has_extra_data:
            image_item['extra_data'] = data_item['extra_data']
        return [tuple((image_item,
                      {'label': data_item[self.label_tag], 'patid': data_item['patid'],
                       'spacing': data_item['spacing'], "num_of_slices": data_item['num_of_slices']}))]


class SeparatePhaseSlice(object):

    def __init__(self):
        self.evaluation_per_phase = True
        self.has_extra_data = False

    def __call__(self, data_item, idx):
        """
        For images and labels we assume the following shape [#slices (z), y, x]
        :param data_item: dictionary. For details see ACDCImageEDES.get_item()
        :param idx:
        :return:
        """
        dict_keys = list(data_item.keys())
        if 'extra_data_es' in dict_keys and 'extra_data_ed' in dict_keys:
            self.has_extra_data = True
        else:
            self.has_extra_data = False

        image_slice_es = data_item["image_es"][idx]
        image_slice_ed = data_item["image_ed"][idx]
        image_item_es = {'image': image_slice_es, 'phase_id': "ES", 'spacing': data_item['spacing'],
                         'patid': data_item['patid'], "slice_id": idx, "num_of_slices": data_item['num_of_slices'],
                         'original_spacing': data_item['original_spacing'], 'frame_id': data_item['frame_id_es'],
                         'is_apex_slice': data_item['apex_base_es']['A'] == idx,
                         'is_base_slice': data_item['apex_base_es']['B'] == idx}
        image_item_ed = {'image': image_slice_ed, 'phase_id': "ED", 'spacing': data_item['spacing'],
                         'patid': data_item['patid'], "slice_id": idx, "num_of_slices": data_item['num_of_slices'],
                         'original_spacing': data_item['original_spacing'], 'frame_id': data_item['frame_id_ed'],
                         'is_apex_slice': data_item['apex_base_ed']['A'] == idx,
                         'is_base_slice': data_item['apex_base_ed']['B'] == idx
                         }
        label_item_es = {'label': data_item["label_es"][idx], 'spacing': data_item['spacing']}
        label_item_ed = {'label': data_item["label_ed"][idx], 'spacing': data_item['spacing']}
        if self.has_extra_data:
            label_item_es['extra_data'] = data_item['extra_data_es'][idx]
            label_item_ed['extra_data'] = data_item['extra_data_ed'][idx]
        return [
            tuple((image_item_es, label_item_es)),
            tuple((image_item_ed, label_item_ed))]


class SeparatePhaseVolume(object):

    def __init__(self, swap_axes=False):
        self.evaluation_per_phase = True
        self.has_extra_data = False
        self.swap_axes = swap_axes

    def __call__(self, data_item):
        dict_keys = list(data_item.keys())
        if 'extra_data_es' in dict_keys and 'extra_data_ed' in dict_keys:
            self.has_extra_data = True
        else:
            self.has_extra_data = False

        image_slice_es = data_item["image_es"]
        image_slice_ed = data_item["image_ed"]

        image_item_es = {'image': image_slice_es, 'phase_id': "ES", 'spacing': data_item['spacing'],
                         'patid': data_item['patid'], "num_of_slices": data_item['num_of_slices'],
                         'original_spacing': data_item['original_spacing'], 'frame_id': data_item['frame_id_es'],
                         'apex_slice_id': data_item['apex_base_es']['A'],
                         'base_slice_id': data_item['apex_base_es']['B'], 'pat_height': data_item['info']['Height'],
                         'pat_weight': data_item['info']['Weight'], 'pat_group': data_item['info']['Group']}
        image_item_ed = {'image': image_slice_ed, 'phase_id': "ED", 'spacing': data_item['spacing'],
                         'patid': data_item['patid'], "num_of_slices": data_item['num_of_slices'],
                         'original_spacing': data_item['original_spacing'], 'frame_id': data_item['frame_id_ed'],
                         'apex_slice_id': data_item['apex_base_ed']['A'],
                         'base_slice_id': data_item['apex_base_ed']['B'], 'pat_height': data_item['info']['Height'],
                         'pat_weight': data_item['info']['Weight'], 'pat_group': data_item['info']['Group']}
        label_item_es = {'label': data_item["label_es"], 'spacing': data_item['spacing']}
        label_item_ed = {'label': data_item["label_ed"], 'spacing': data_item['spacing']}
        if self.has_extra_data:
            label_item_es['extra_data'] = data_item['extra_data_es']
            label_item_ed['extra_data'] = data_item['extra_data_ed']
        return [
            tuple((image_item_es, label_item_es)),
            tuple((image_item_ed, label_item_ed))]


class CompleteCardiacCycleVolume(object):

    def __init__(self, pad_size):
        self.pad_length = pad_size

    def __call__(self, data_item):
        image = data_item['image']
        # image has shape [#phases, z, y, x] but for DCNN models we need shape [#phases, z, 1, y, x]
        # we are still dealing with numpy arrays
        image = np.expand_dims(image, axis=2)
        # [io_channels, w, h, d]
        image = np.pad(image,
                       ((0, 0), (0, 0), (0, 0), (self.pad_length, self.pad_length),
                        (self.pad_length, self.pad_length)),
                       'constant',
                       constant_values=(0,)).astype(np.float32)
        image = torch.FloatTensor(torch.from_numpy(image).float())
        sample = {'image': image}
        other_keys = set(data_item.keys()) - set(['image'])
        sample.update(dict((k, data_item[k]) for k in other_keys))
        return sample


class RandomRotation(object):
    def __init__(self, rs, only_inplane=True):
        self.rs = rs
        self.only_inplane = only_inplane

    def __call__(self, data_tuple):
        if self.only_inplane:
            return self.inplane(data_tuple)
        else:
            raise NotImplementedError

    def inplane(self, data_tuple):
        data_image = data_tuple[0]
        data_label = data_tuple[1]
        image, label = data_image['image'], data_label['label']
        rs = self.rs
        k = rs.randint(4)
        # without .copy() I got later pytorch error when casting from numpy to torch.tensor
        # "ValueError: some of the strides of a given numpy array are negative."
        if image.ndim == 2:
            image = np.rot90(image, k, (0, 1)).copy()
            label = np.rot90(label, k, (0, 1)).copy()
        elif image.ndim == 3:
            # [io_channels, x, y]
            image = np.rot90(image, k, (1, 2)).copy()
            label = np.rot90(label, k, (1, 2)).copy()
        else:
            raise ValueError("ERROR - RandomRotation - image rank not supported")
        if 'extra_data' in list(data_tuple[1].keys()):
            extra_data = data_tuple[1]['extra_data']
            if extra_data.ndim == 3:
                extra_data = np.rot90(extra_data, k, (1, 2)).copy()
            elif extra_data.ndim == 4:
                # [io_channels, x, y]
                extra_data = np.rot90(extra_data, k, (2, 3)).copy()
            data_tuple[1]['extra_data'] = extra_data

        data_tuple[0]['image'] = image
        data_tuple[1]['label'] = label

        return data_tuple


class RandomMirroring(object):
    def __init__(self, rs):
        self.rs = rs

    def __call__(self, data_tuple):
        image = data_tuple[0]['image']
        label = data_tuple[1]['label']
        rs = self.rs
        if rs.randint(2):
            if image.ndim == 2:
                image = image[::-1]
                image = image[:, ::-1]
                label = label[::-1]
                label = label[:, ::-1]
            elif image.ndim == 3:
                # [io_channels, x, y]
                image = image[:, ::-1]
                image = image[:, :, ::-1]
                label = label[:, ::-1]
                label = label[:, :, ::-1]
            else:
                raise ValueError("ERROR - RandomMirroring - image rank not supported")
            if 'extra_data' in list(data_tuple[1].keys()):
                extra_data = data_tuple[1]['extra_data']
                if extra_data.ndim == 3:
                    # currently extra_data = dt_maps: so we have shape [nclasses, y, x]
                    extra_data = extra_data[:, ::-1]
                    extra_data = extra_data[:, :, ::-1]

                elif extra_data.ndim == 4:
                    # currently extra_data = dt_maps: so we have shape [z, nclasses, y, x]
                    extra_data = extra_data[:, :, ::-1]
                    extra_data = extra_data[:, :, :, ::-1]
                else:
                    raise ValueError("ERROR - RandomMirroring - label['extra_data'] rank not supported")

                data_tuple[1]['extra_data'] = extra_data.copy()

        data_tuple[0]['image'] = image.copy()
        data_tuple[1]['label'] = label.copy()
        return data_tuple


class CenterCrop(object):

    def __init__(self, dilated_psize, psize):

        self.dilated_psize = dilated_psize
        self.psize = psize

    def __call__(self, data_tuple):

        data_image = data_tuple[0]
        data_label = data_tuple[1]
        image = data_image['image']
        current_shape = RandomCrop._make_tuple(image.shape)
        offx, offy = self._get_img_offsets(current_shape)

        image = image[..., offx:offx + self.dilated_psize, offy:offy + self.dilated_psize]
        label = data_label['label']
        label = label[..., offx:offx + self.psize + 1, offy:offy + self.psize + 1]
        # if multi_labels is not None:
        #    multi_labels = multi_labels[..., offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]

        data_tuple[0]['image'] = image
        data_tuple[1]['label'] = label
        return data_tuple

    def _get_img_offsets(self, img_shape):

        w, h = img_shape
        offx = int((w - self.dilated_psize) / 2)
        offy = int((h - self.dilated_psize) / 2)
        return offx, offy


class RandomCrop(object):

    def __init__(self, dilated_psize, psize, rs):
        self.rs = rs
        self.dilated_psize = dilated_psize
        self.psize = psize

    @staticmethod
    def _make_tuple(in_shape):
        # we assume that in_shape is a tuple (indicating shape) of 2 or more dimensions. The last 2 dims are always w, h
        return tuple((in_shape[-2], in_shape[-1]))

    def __call__(self, data_tuple):

        data_image = data_tuple[0]
        data_label = data_tuple[1]
        image = data_image['image']
        current_shape = RandomCrop._make_tuple(image.shape)
        offx, offy = self._get_img_offsets(current_shape)
        image = image[..., offx:offx + self.dilated_psize, offy:offy + self.dilated_psize]
        label = data_label['label']
        label = label[..., offx:offx + self.psize + 1, offy:offy + self.psize + 1]
        if 'extra_data' in list(data_label.keys()):
            data_label['extra_data'] = data_label['extra_data'][..., offx:offx + self.psize + 1, offy:offy + self.psize + 1]

        data_tuple[0]['image'] = image
        data_tuple[1]['label'] = label
        return data_tuple

    def _get_img_offsets(self, img_shape):

        w, h = img_shape
        offx = self.rs.choice(np.arange(w - self.dilated_psize), 1, replace=True)[0]
        offy = self.rs.choice(np.arange(h - self.dilated_psize), 1, replace=True)[0]

        return offx, offy


class DCNNPadding(object):

    def __init__(self, pad_size, is_volume=False):
        self.pad_length = pad_size
        self.is_volume = is_volume
        self.has_extra_data = False

    def __call__(self, data_tuple):
        if self.is_volume:
            return self._pad_volume(data_tuple)
        else:
            return self._pad_slice(data_tuple)

    def _pad_slice(self, data_tuple):
        image = data_tuple[0]['image']

        if image.ndim == 3:
            # [io_channels, w, h]
            image = np.pad(image,
                          ((0, 0), (self.pad_length, self.pad_length),
                           (self.pad_length, self.pad_length)),
                           'constant',
                           constant_values=(0,)).astype(np.float32)
        elif image.ndim == 2:
            image = np.pad(image, self.pad_length, 'constant',
                           constant_values=(0,)).astype(np.float32)
        else:
            raise ValueError("ERROR - format_dcnn_batch_item - array has unsupported rank {}".format(image.ndim))

        data_tuple[0]['image'] = image
        return data_tuple

    def _pad_volume(self, data_tuple):
        image = data_tuple[0]['image']
        if image.ndim == 4:
            # [io_channels, d, w, h] OR Z, Y, X !!!
            image = np.pad(image,
                          ((0, 0), (0, 0), (self.pad_length, self.pad_length),
                           (self.pad_length, self.pad_length)),
                           'constant',
                           constant_values=(0,)).astype(np.float32)

        elif image.ndim == 3:
            # [z, y, x]
            image = np.pad(image,
                          ((0, 0), (self.pad_length, self.pad_length),
                           (self.pad_length, self.pad_length)),
                           'constant',
                           constant_values=(0,)).astype(np.float32)
        else:
            raise ValueError("ERROR - format_dcnn_batch_item - array has unsupported rank {}".format(image.ndim))
        data_tuple[0]['image'] = image

        return data_tuple


class LabelBinarizer(object):

    def __init__(self, num_classes=4, keep_multi_labels=False):
        self.num_classes = num_classes
        self.keep_multi_labels = keep_multi_labels

    def __call__(self, data_tuple):

        label = data_tuple[1]['label']
        if self.num_classes != 4 and label.ndim == 2:
            raise ValueError("ERROR - LabelBinarizer - #classes is {} but dim0 of label is not 2 "
                             "(combined ES/ED".format(self.num_classes))

        data_tuple[1]['label'] = self._binarize_labels(label)
        if self.keep_multi_labels:
            data_tuple[1]['multi_label'] = label
        else:
            data_tuple[1]['multi_label'] = np.ones((1, 1, 1))

        return data_tuple

    def _binarize_labels(self, labels):
        if self.num_classes == 4:
            return one_hot_encoding(labels)
        else:
            # we know that the first dimension of labels is 2, because we only end up here if we have combined
            # ES and ED. Hence we can make a new shape omitting the first dim
            new_shape = tuple((self.num_classes,) + labels.shape[1:])
            new_labels = np.zeros(new_shape)
            new_labels[:int(self.num_classes / 2)] = one_hot_encoding(labels[0])
            new_labels[int(self.num_classes / 2):] = one_hot_encoding(labels[1])
            return new_labels


class ToTensor(object):

    def __init__(self, is_volume=False):
        self.is_volume = is_volume
        self.has_extra_data = False
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data_tuple):
        """

        :param data_tuple: 1) dictionary for image 2) dictionary for label
        :return:
        """
        if 'extra_data' in list(data_tuple[1].keys()):
            self.has_extra_data = True
        else:
            self.has_extra_data = False
        if self.is_volume:
            return self._volume(data_tuple)
        else:
            return self._slices(data_tuple)

    def _slices(self, data_tuple):
        data_image = data_tuple[0]
        data_label = data_tuple[1]
        image, label, spacing = data_image['image'], data_label['label'], data_image['spacing']
        multi_label = data_label['multi_label']
        if image.ndim == 2:
            data_tuple[0]['image'] = torch.unsqueeze(torch.FloatTensor(torch.from_numpy(image).float()), 0)
        else:
            data_tuple[0]['image'] = torch.FloatTensor(torch.from_numpy(image).float())

        data_tuple[1]['label'] = torch.ByteTensor(torch.from_numpy(label).byte())
        if self.has_extra_data:
            # currently extra_data = dt_maps which has shape [z, nclasses, x, y] but if we're processing slices the
            # z-dim is missing and hence, we need to add a new batch dimension
            data_label['extra_data'] = torch.FloatTensor(torch.from_numpy(data_label['extra_data']).float())

        if multi_label is not None:
            data_tuple[1]['multi_label'] = torch.ByteTensor(torch.from_numpy(multi_label).byte())

        return data_tuple

    def _volume(self, data_tuple):
        # for volumes we want to return shape [d, C, w, h] where d is actually the batch size N but the number
        # of slices of the volume.
        # images and labels for the volumes have shape [z, y, x] !!!
        # labels have shape [nclasses, z, y, x]
        data_image = data_tuple[0]
        data_label = data_tuple[1]

        image, label, spacing = data_image['image'], data_label['label'], data_image['spacing']
        multi_label = data_label['multi_label']

        if image.ndim == 4:
            # E.g. when using ESEDCombined dataset we end up here with image.shape [2, d, w, h], so we only
            # swap axis d to the front
            image = torch.FloatTensor(torch.from_numpy(image).float())
        elif image.ndim == 3:
            # shape is [d, w, h]. We first add a dummy dimension in front and then transpose to [1, d, w, h]
            image = torch.unsqueeze(torch.FloatTensor(torch.from_numpy(image).float()), 0)
        else:
            raise ValueError("ERROR - ToTensor._volume: unsupported rank of image {}".format(image.ndim))

        # So we assume image has now shape [io_channels, z, y, x]. We need to swap z and io_channels
        data_tuple[0]['image'] = image.permute(1, 0, 2, 3)
        label = torch.ByteTensor(torch.from_numpy(label).byte())
        # again label has already shape [nclasses, z, y, x] we need to swap z and nclasses
        data_tuple[1]['label'] = label.permute(1, 0, 2, 3)
        if self.has_extra_data:
            # currently extra_data = dt_maps which has shape [z, nclasses, x, y]
            if data_label['extra_data'].ndim == 4:
                data_label['extra_data'] = torch.FloatTensor(
                    torch.from_numpy(data_label['extra_data']).float())
            else:
                raise ValueError("ERROR - rank {} is not supported (only 3/4)".format(data_label['extra_data'].ndim))

        if multi_label is not None:
            multi_label = torch.ByteTensor(torch.from_numpy(multi_label).byte())
            if multi_label.dim() == 3:
                # has shape [w, h, d]
                data_tuple[1]['multi_label'] = multi_label.permute(2, 0, 1)
            elif multi_label.dim() == 4:
                # has shape [2, w, h, d]
                data_tuple[1]['multi_label'] = multi_label.permute(3, 0, 1, 2)
            else:
                raise ValueError("ERROR - ToTensor._volume: unsupported rank of multi_label "
                                 "{}".format(multi_label.dim()))

        return data_tuple
