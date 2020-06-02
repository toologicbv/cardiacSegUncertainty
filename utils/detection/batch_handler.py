import os
import torch
import numpy as np

import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from plotting.color_maps import transparent_cmap
from plotting.color_maps import SegColorMap
from utils.box_utils import BoundingBox
# from utils.common import convert_to_multiclass
from networks.detection.general_setup import config_detector


def create_grid_heat_map(pred_probs, grid_spacing, w, h, prob_threshold=0.):
    """

    :param pred_probs: has shape [w, h] NOT FLATTENED
    :param grid_spacing:
    :param w:
    :param h:

    :param prob_threshold:
    :return: heat-map of original patch or image size. we fill the grid-blocks with the values from the predicted
                        probabilities.
    """
    heat_map = np.zeros((w, h))
    grid_map_x, grid_map_y = [], []
    grid_spacings_w = np.arange(0, w + grid_spacing, grid_spacing)[1:]
    grid_spacings_h = np.arange(0, h + grid_spacing, grid_spacing)[1:]
    # Second split label slice horizontally
    start_w = 0
    pred_idx_w = 0
    for w_offset in grid_spacings_w:
        pred_idx_h = 0
        start_h = 0
        for h_offset in grid_spacings_h:
            if pred_probs is not None:
                block_prob = pred_probs[pred_idx_w, pred_idx_h]
            else:
                block_prob = 0
            if block_prob >= prob_threshold:
                heat_map[start_w:w_offset, start_h:h_offset] = block_prob
            grid_map_x.append((start_w + w_offset) / 2.)
            grid_map_y.append((start_h + h_offset) / 2.)
            start_h = h_offset
            pred_idx_h += 1
        start_w = w_offset
        pred_idx_w += 1
    # print("grid_map_x {} grid_map_y {}".format(len(grid_map_x), len(grid_map_y)))
    # print("pred_idx_w {} pred_idx_h {}".format(pred_idx_w, pred_idx_h))
    return heat_map, [np.array(grid_map_x), np.array(grid_map_y)], None


class BatchHandler(object):

    file_suffix = "_heat_map"
    mri_index = 0
    uncertainty_index = 1
    auto_mask_index = 2

    def __init__(self, data_set, is_train=True, cuda=True, input_channels="allchannels", keep_bounding_boxes=False,
                 num_of_max_pool_layers=None, app_config=config_detector, verbose=False,
                 fixed_roi_center=False):
        """

        :param data_set:
        :param is_train:
        :param cuda:
        :param verbose:
        :param keep_bounding_boxes:
        :param num_of_max_pool_layers:
        :param patch_size: list with two values e.g. [80, 80]. Default is provided in config object
        :param app_config: default config_detector object
        :param fixed_roi_center: boolean: If true during training we don't randomly sample a center point inside the
                                 target labels (for debugging purposes)
        """
        """
            data_set  of object type SliceDetectorDataSet
            patch_size needs to be numpy array [w, h]
        """
        self.app_config = app_config
        self.fixed_roi_center = fixed_roi_center
        self.max_grid_spacing = 8
        if is_train:
            self.patch_size = self.app_config.patch_size
        else:
            self.patch_size = self.app_config.test_patch_size
        self.cuda = cuda
        self.is_train = is_train
        self.number_of_patients = data_set.get_size(is_train=is_train)
        self.num_of_channels = 3  # data_set.num_of_channels
        # 0=mri         1=uncertainties         2=labels    IF NONE then we use all channels
        if input_channels == "allchannels":
            self.input_channels = None
        elif input_channels == "mronly":
            self.input_channels = 0
        elif input_channels == "umap" or input_channels == "bmap" or input_channels == "emap":
            self.input_channels = 1
        elif input_channels == "segmask":
            self.input_channels = 2

        self.data_set = data_set

        self.batch_dta_slice_ids = []
        self.sample_range = self.data_set.get_size(is_train=is_train)
        self.verbose = verbose
        self.batch_bounding_boxes = None
        self.keep_bounding_boxes = keep_bounding_boxes
        self.batch_label_slices = []
        self.batch_size = 0
        self.batch_images = None
        self.batch_labels_per_voxel = None
        self.batch_patient_ids = []
        # useful in order to find back the original patient+slice IDs /  only used for test-batch
        self.batch_patient_slice_id = []
        # stores indications for base/apex=1 or middle=0
        self.batch_extra_labels = None
        self.target_labels_per_roi = None
        # holds the number of incorrectly segmented voxels per patch
        self.target_labels_stats_per_roi = None
        self.target_labels_regression = None
        # keep batch items
        self.keep_batch_images = []
        self.keep_batch_target_labels = []
        self.keep_batch_label_slices = []
        self.keep_batch_target_counts = []
        # during testing, we skip slices that do not contain any automatic segmentations
        self.num_of_skipped_slices = 0
        self.default_patch = 0
        # when calling with keep_batch=True, we also want to store the predicted probs during testing
        # only during eval time
        self.batch_pred_probs = None
        self.batch_pred_labels = None
        self.batch_gt_labels = None
        # dictionary with key=slice_id and value list of indices that refer to the numpy array indices
        # of batch_pred_probs
        # and batch_pred_labels, batch_gt_labels. we need these to evaluate the performance PER SLICE (FROC curve)
        # e.g.
        self.batch_slice_pred_probs = {}
        self.batch_slice_recon_map = {}
        self.batch_slice_gt_labels = {}
        # same as above but then separately for each patient:
        self.batch_patient_pred_probs = {}
        self.batch_patient_gt_labels = {}
        self.batch_patient_gt_label_counts = {}
        # in order to link slice_ids from the dataset used to generate batches, to the patient_id (for analysis only)
        self.trans_dict = {}
        # we store the last slice_id we returned for the test batch, so we can chunk the complete test set if
        # necessary. Remember this is an index referring to the slices in list dataset.test_images
        self.last_test_list_idx = None
        if num_of_max_pool_layers is None:
            self.num_of_max_pool_layers = 3
        else:
            self.num_of_max_pool_layers = num_of_max_pool_layers

    def add_probs(self, probs, slice_id=None):
        """

        :param probs: has shape [1, w, h] and contains only the probabilities for the positive(true) class
        :param slice_id:
        :return: None
        """
        self.batch_pred_probs.append(probs)
        if slice_id is not None:
            self.batch_slice_pred_probs[slice_id] = probs.flatten()

    def add_reconstruction(self, recon_map, patient_id, cardiac_phase, slice_id, patch_slice_xy):
        """

        :param patient_id
        :param recon_map: has shape [1, w, h] and contains only the probabilities for the positive(true) class
        :param cardiac_phase
        :param slice_id: NOTE: runs from 1...num of slices (IMPORTANT: NEED TO SUBSTRACT ONE)
        :return: None
        """

        slice_id -= 1
        if recon_map.ndim > 2:
            # squeeze batch dimension (we use this during testing and numpy array has probably shape [1, w, h]
            recon_map = np.squeeze(recon_map, axis=0)

        if patient_id not in list(self.batch_slice_recon_map.keys()):
            # we get the original image size for this patient from the data set. trans_dict contains per patient
            # a tuple(is_train, [image idx]). Second is a list of image indices that we can use to pick one example
            # for this patient. Slices for this patient all have the same shape. Remember input image for detector
            # have shape [3, w, h], so we skip the first dim
            is_train, data_set_idx = self.data_set.trans_dict[patient_id]
            # first get the # of slices for this patient. We need to construct the empty numpy array for the
            # reconstructions
            # NOTE: the number is for ONE cardiac phase. So we need to double
            if is_train:
                _, w, h = self.data_set.train_images[data_set_idx[0]].shape
                num_of_slices = self.data_set.train_patient_num_of_slices[patient_id]
            else:
                _, w, h = self.data_set.test_images[data_set_idx[0]].shape
                num_of_slices = self.data_set.test_patient_num_of_slices[patient_id]

            self.batch_slice_recon_map[patient_id] = np.zeros((2, w, h, num_of_slices))

        self.batch_slice_recon_map[patient_id][cardiac_phase, patch_slice_xy[0], patch_slice_xy[1], slice_id] = recon_map

    def add_gt_labels_slice(self, gt_labels, slice_id):
        self.batch_slice_gt_labels[slice_id] = gt_labels

    def flatten_batch_probs(self):
        # Only used during testing
        # batch_pred_probs is a list of numpy arrays with shape [1, w_grid, h_grid]. For our analysis of the model
        # performance we flatten here all probs and concatenate them into ONE np array
        flattened_probs = np.empty(0)
        for probs in self.batch_pred_probs:
            flattened_probs = np.concatenate((flattened_probs, probs.flatten()))
        return flattened_probs

    def fill_trans_dict(self):
        self.trans_dict = {}
        # only fill the translation dictionary (slice_id -> patient_id) for the slices we are processing in this batch
        for slice_id in self.batch_dta_slice_ids:
            for patient_id, slice_info in self.data_set.trans_dict.items():
                # slice_info is a 2-tuple with is_train indication and list of slice ids
                # we're only interested in train or test images. depends on the batch we're evaluating
                if slice_info[0] == self.is_train:
                    if slice_id in slice_info[1]:
                        self.trans_dict[slice_id] = patient_id
                        break

    def add_pred_labels(self, pred_labels):
        self.batch_pred_labels.extend(pred_labels)

    def add_gt_labels(self, gt_labels):
        self.batch_gt_labels.extend(gt_labels)

    def add_patient_slice_results(self, patient_id, slice_id, cardiac_phase, pred_probs, gt_labels, gt_voxel_counts):
        slice_id -= 1
        if patient_id not in list(self.batch_patient_gt_labels.keys()):
            self.batch_patient_gt_labels[patient_id] = {cardiac_phase: {}}
            self.batch_patient_gt_label_counts[patient_id] = {cardiac_phase: {}}
            self.batch_patient_pred_probs[patient_id] = {cardiac_phase: {}}
        else:
            if cardiac_phase not in list(self.batch_patient_gt_labels[patient_id].keys()):
                self.batch_patient_gt_labels[patient_id][cardiac_phase] = {}
                self.batch_patient_gt_label_counts[patient_id][cardiac_phase] = {}
                self.batch_patient_pred_probs[patient_id][cardiac_phase] = {}

            # patient as key already exists
        self.batch_patient_gt_labels[patient_id][cardiac_phase][str(slice_id)] = gt_labels.flatten()
        self.batch_patient_gt_label_counts[patient_id][cardiac_phase][str(slice_id)] = gt_voxel_counts.flatten()
        self.batch_patient_pred_probs[patient_id][cardiac_phase][str(slice_id)] = pred_probs.flatten()

    def __call__(self, batch_size=None, do_balance=True, keep_batch=False, disrupt_chnnl=None):
        """
        Construct a batch of shape [batch_size, 3channels, w, h]
                                   3channels: (1) input image
                                              (2) generated (raw/unfiltered) u-map/entropy map
                                              (3) predicted segmentation mask

        :param batch_size:
        :param disrupt_chnnl:  integer {0, 1, 2}: see meaning of channels above

        :param do_balance: if True batch is balanced w.r.t. slices containing positive target areas and not (1:3)
        :param keep_batch: boolean, only used during TESTING. Indicates whether we keep the batch items in
                           self.batch_images, self.target_labels_per_roi (lists)
        :return: input batch and corresponding references
        """
        self.batch_bounding_boxes = np.zeros((batch_size, 4))
        self.batch_label_slices = []
        self.keep_batch_images = []
        self.keep_batch_target_labels = []
        self.keep_batch_label_slices = []
        self.batch_slice_areas = []
        # stores 1=apex/base or 0=middle slice
        self.batch_extra_labels = []
        self.batch_patient_slice_id = []
        self.batch_dta_slice_ids = []
        self.target_labels = None
        self.batch_size = batch_size
        if self.is_train:
            # (1) batch_images will have the shape [#batches, num_of_input_chnls, w, h]
            # (2) batch_labels_per_voxel: Contains the same info as the original filtered seg-errors. We use these
            #                             for training of the rd4 model with deconvolutions to generate voxel-heat-maps
            # (3) target_labels_per_roi: is a dictionary (key batch-item-nr) of dictionaries (key grid spacing)
            self.batch_images, self.batch_labels_per_voxel, self.target_labels_per_roi = \
                self._generate_train_batch(batch_size, do_balance=do_balance, keep_batch=keep_batch)

            if self.cuda:
                self.batch_images = self.batch_images.cuda()
                for g_spacing, target_lbls in self.target_labels_per_roi.items():
                    # we don't use grid-spacing "1" key (overall binary indication whether patch contains target
                    # voxels (pos/neg example)
                    if g_spacing != 1:
                        target_lbls = target_lbls.cuda()
                        self.target_labels_per_roi[g_spacing] = target_lbls
            # IMPORTANT: self.batch_images has [batch_size, 3, x, y]
            if self.input_channels is not None:
                if self.input_channels == 0:
                    self.batch_images = self.batch_images[:, self.input_channels].unsqueeze(1)
                else:
                    self.batch_images = torch.cat((self.batch_images[:, 0].unsqueeze(1),
                                                  self.batch_images[:, self.input_channels].unsqueeze(1)), dim=1)
            return self.batch_images, self.target_labels_per_roi
        else:
            # during testing we process whole slices and loop over the patient_ids in the test set.
            return self._generate_test_batch(batch_size, keep_batch=keep_batch, disrupt_chnnl=disrupt_chnnl)

    def _get_center_roi(self, t_labels):
        """
        In some cases (debugging) we use the center of the auto segmentation mask (below in method _adjust_roi_to_patch
        we sample a position somewhere in the roi area).
        Returns tuple of slice_x and slice_y

        :param t_labels: [w, h] a slice containing the target rois for detection
        :return:
        """
        if t_labels.ndim == 3:
            # in case we use 3d input for the model in the first layer, and the middle slice contains (target)
            # contains no positive labels (voxels to be inspected) t_labels contains the automatic segmentation mask
            # which is part of the input (third channel). In this case t_labels has shape [depth, w, h] and we only
            # want the middle slice (depth=3 as well). So, we take index 1 of the first dimension
            t_labels = t_labels[1]

        half_width = int(self.patch_size[0] / 2)
        half_height = int(self.patch_size[1] / 2)
        w, h = t_labels.shape
        img_half_width = int(w / 2)
        img_half_height = int(h / 2)
        slice_x = slice(img_half_width - half_width,
                        img_half_width + half_width, None)
        slice_y = slice(img_half_height - half_height,
                        img_half_height + half_height, None)
        return tuple((slice_x, slice_y))

    def _adjust_roi_to_patch(self, t_labels, slice_num):
        """
        we need to make sure the roi area is not too close to image boundaries, because we will sample a center
        coordinate from the roi to build the patch around it.
        :param t_labels: binary numpy array [w, h] containing target voxels (or auto seg mask when no target labels
                         are present in the selected image slice)
        :param slice_num:
        :return: slice_x, slice_y for patch
        """
        if t_labels.ndim == 3:
            # in case we use 3d input for the model in the first layer, and the middle slice contains (target)
            # contains no positive labels (voxels to be inspected) t_labels contains the automatic segmentation mask
            # which is part of the input (third channel). In this case t_labels has shape [depth, w, h] and we only
            # want the middle slice (depth=3 as well). So, we take index 1 of the first dimension
            t_labels = t_labels[1]

        half_width = int(self.patch_size[0] / 2)
        half_height = int(self.patch_size[1] / 2)
        w, h = t_labels.shape
        t_indices = np.where(t_labels != 0)
        num_of_indices = t_indices[0].shape[0]
        # t_labels has at least one positive voxel. t_indices contains of two arrays, for x-axis idx, for y-axis idx
        do_continue = True
        num_of_iters = 0

        while do_continue:
            idx = np.random.randint(0, num_of_indices, size=1, dtype=np.int)[0]
            x = t_indices[0][idx]
            y = t_indices[1][idx]
            if x + half_width > w or x - half_width < 0:
                if x + half_width > w:
                    # to close to the right border
                    x_start, x_stop = w - self.patch_size[0], w
                else:
                    # to close to the left border
                    x_start, x_stop = 0, self.patch_size[0]
            else:
                x_start, x_stop = x - half_width, x + half_width

            if y + half_height > h or y - half_height < 0:
                if y + half_height > h:
                    # to close to the lower border
                    y_start, y_stop = h - self.patch_size[1], h
                else:
                    # to close to the left border
                    y_start, y_stop = 0, self.patch_size[1]
            else:
                y_start, y_stop = y - half_width, y + half_width

            slice_x = slice(x_start, x_stop, None)
            slice_y = slice(y_start, y_stop, None)
            # if slice_num in [492, 495, 493]:
            #    print("WARNING - BatchHandler - ", slice_num, slice_x, slice_y)
            do_continue = False
            num_of_iters += 1

        return tuple((slice_x, slice_y))

    def _generate_train_batch(self, batch_size, do_balance=True, keep_batch=False):
        """

        :param batch_size:
        :param do_balance: if set to True, batch is separated between negative and positive examples based on
                            fraction_negatives (currently 2/3)
        :return: (1) batch_imgs: torch.FloatTensors [batch_size, 3, patch_size, patch_size]
                 (2) np_batch_lbls: binary numpy array [batch_size, patch_size, patch_size]
                 (3) target_labels: dictionary (key batch-item-nr) of dictionary torch.LongTensors.
                                                                keys 1, 4, 8 (currently representing grid spacing)
        """
        num_of_negatives = int(batch_size * self.app_config.fraction_negatives)
        num_of_positives = batch_size - num_of_negatives
        negative_idx = []
        positive_idx = []
        batch_imgs = torch.zeros((batch_size, self.num_of_channels, self.patch_size[0],
                                      self.patch_size[1]))
        # this dictionary only contains the binary indications for each batch item (does contain voxels or not)
        # contains currently three key values that represent the grid spacing:
        #                       1 = complete patch, we don't use this
        #                       4 = after maxPool 2, 4x4 patches hence, grid spacing 4
        #                       8 = after maxPool 3, 8x8 patches hence, grid spacing 8
        # Each dict entry contains torch.LongTensor array with binary indications
        target_labels = {1: np.zeros(batch_size)}
        # store the number of positive gt labels per grid. we use this to analyze the errors the model makes
        self.target_labels_stats_per_roi = {1: np.zeros(batch_size)}
        for i in np.arange(2, self.num_of_max_pool_layers + 1):
            grid_spacing = int(2 ** i)
            g = int(self.patch_size[0] / grid_spacing)
            target_labels[grid_spacing] = torch.zeros((batch_size, g * g), dtype=torch.long)
            self.target_labels_stats_per_roi[grid_spacing] = np.zeros((batch_size, g * g))
        # this array holds the patch for each label_slice (binary). Is a numpy array, we only need this object
        # to determine the final target labels per grid/roi
        np_batch_lbls = np.zeros((batch_size, self.patch_size[0], self.patch_size[1]))
        max_search_iters = batch_size * 10
        cannot_balance = False
        do_continue = True

        # in case the target area is too close to the boundaries of the image (we create a patch around the target area)
        # then we just choose the center location of the image and create a batch around. Should not happen often
        # only a couple of times when we use the raw segmentation errors as supervision of the task
        self.default_patch = 0
        i = 0
        # start with collecting image slice indices
        while do_continue:
            slice_num = np.random.randint(0, self.sample_range, size=1, dtype=np.int)[0]
            # IMPORTANT: np array train_lbl_rois contains filtered target rois for supervision
            # train_lbl_rois has shape [N,4] and contains box_four notation for target areas
            # IMPORTANT: train_pred_lbl_rois contains one array with bounding box specifications
            # train_pred_lbl_rois
            num_of_target_rois = self.data_set.train_lbl_rois[slice_num].shape[0]
            pred_lbls_exist = (np.sum(self.data_set.train_pred_lbl_rois[slice_num]) != 0).astype(np.bool)
            if num_of_target_rois == 0 and not pred_lbls_exist:
                # print("WARNING - No automatic mask and no seg-errors - hence continue")
                # TODO: 13-03-2019 we changed this. currently training with these!!
                pass
                # continue
            i += 1
            # we enforce the ratio between positives-negatives up to a certain effort (max_search_iters)
            # and when we indicated that we want to balance anyway
            # Logic: (1) if we still need to collect positives but the slice does not contain target rois
            #            and we collected already enough negatives, then we skip this slice.
            #        (2) or: we already have enough positives and but enough negatives and this slice contains
            #            target rois, we will skip this slice.
            if do_balance and (num_of_positives != 0 and num_of_target_rois == 0 and \
                               num_of_negatives == 0 and i <= max_search_iters) or \
                    (num_of_positives == 0 and num_of_target_rois != 0 and \
                     num_of_negatives != 0 and i <= max_search_iters):
                # no positive target area in slice and we already have enough negatives. Hence, continue
                if self.verbose:
                    print(("Continue - #negatives {} out of {}".format(num_of_negatives, len(negative_idx))))
                continue
            elif not do_balance:
                # we don't need to bother because WE ARE NOT BALANCING THE BATCH (between neg/pos examples)
                pass
            else:
                if i > max_search_iters and not cannot_balance:
                    cannot_balance = True
                    # print("WARNING - Reached max-iters {}".format(i))
            # We need to make sure that if we still need negatives, and we don't have target rois to detect
            # then we AT LEAST need predicted labels. It's one or the other. In case we didn't predict anything
            # AND we didn't make any errors, then skip this slice
            if do_balance and num_of_target_rois == 0 and not pred_lbls_exist and i <= max_search_iters \
                    and num_of_negatives != 0:
                continue
            # this is for the case WE DON't balance. Make sure pred_lbls_exist = TRUE, otherwise continue
            elif not do_balance and not pred_lbls_exist and i <= max_search_iters:
                continue
            elif not do_balance and num_of_target_rois == 0 and not pred_lbls_exist:
                print(("WARNING - {} <= max_search_iters ".format(i)))

            if num_of_target_rois == 0:
                num_of_negatives -= 1
                # no voxels in target label slice, hence, we take the automatic seg-mask (channel 2 of input)
                # to select a center voxel for the patch. If the automatic seg mask is empty (no predictions at all)
                # we take the uncertainty map that determines the crop region
                if pred_lbls_exist:
                    channel = 2  # auto mask
                else:
                    channel = 1  # uncertainty map
                    print("WARNING - using uncertainty map to determine crop region")

                if self.fixed_roi_center:
                    patch_slice_xy = self._get_center_roi(self.data_set.train_images[slice_num][channel])
                else:
                    patch_slice_xy = self._adjust_roi_to_patch(self.data_set.train_images[slice_num][channel], slice_num)
                negative_idx.append(tuple((slice_num, patch_slice_xy)))
            else:
                if self.verbose:
                    print(("#Positives {}".format(num_of_target_rois)))
                num_of_positives -= 1
                # we have a slice with positive target voxels. Hence we use the target slice to find a center voxel
                # that we use to build the batch patch.
                # The method returns a tuple of (slice_x, slice_y) with which we can build the patch further on
                if self.fixed_roi_center:
                    patch_slice_xy = self._get_center_roi(self.data_set.train_labels[slice_num])
                else:
                    patch_slice_xy = self._adjust_roi_to_patch(self.data_set.train_labels[slice_num], slice_num)
                positive_idx.append(tuple((slice_num, patch_slice_xy)))

            base_apex_slice = self.data_set.train_labels_extra[slice_num]
            self.batch_slice_areas.append(patch_slice_xy)
            self.batch_extra_labels.append(base_apex_slice)
            self.batch_patient_slice_id.append(self.data_set.train_patient_slice_id[slice_num])

            if len(positive_idx) + len(negative_idx) == batch_size:
                do_continue = False
        if cannot_balance:
            print("WARNING - Negative/positive ratio {}/{}".format(len(negative_idx), len(positive_idx)))
        b = 0
        for slice_area_spec in positive_idx:
            # print("INFO - Positives slice num {}".format(slice_area_spec))
            batch_imgs[b], np_batch_lbls[b], overall_lbls = self._create_train_batch_item(slice_area_spec,
                                                                                             is_positive=True,
                                                                                             batch_item_nr=b)
            target_labels, self.target_labels_stats_per_roi = self._generate_batch_labels(np_batch_lbls[b], target_labels,
                                                                                          batch_item_nr=b,
                                                                                          is_positive=overall_lbls,
                                                        target_labels_stats_per_roi=self.target_labels_stats_per_roi)
            target_labels[1][b] = overall_lbls
            self.batch_dta_slice_ids.append(slice_area_spec[0])
            if keep_batch:
                self.keep_batch_images.append(batch_imgs[b])
                keep_label_grids = {}
                for grid_key in list(target_labels.keys()):
                    keep_label_grids[grid_key] = target_labels[grid_key][b]
                self.keep_batch_target_labels.append(keep_label_grids)
                self.keep_batch_label_slices.append(np_batch_lbls[b])
            b += 1

        for slice_area_spec in negative_idx:
            # print("INFO - Negatives slice num {}".format(slice_area_spec))
            # _create_train_batch_item returns: torch.FloatTensor [3, patch_size, patch_size],
            #                                   Numpy array [patch_size, patch_size]
            #                                   Binary scalar value indicating pos/neg slice
            batch_imgs[b], np_batch_lbls[b], overall_lbls = self._create_train_batch_item(slice_area_spec,
                                                                                          is_positive=False,
                                                                                          batch_item_nr=b)
            target_labels[1][b] = overall_lbls
            self.batch_dta_slice_ids.append(slice_area_spec[0])
            if keep_batch:
                self.keep_batch_images.append(batch_imgs[b])
                keep_label_grids = {}
                for grid_key in list(target_labels.keys()):
                    keep_label_grids[grid_key] = target_labels[grid_key][b]
                self.keep_batch_target_labels.append(keep_label_grids)
                self.keep_batch_label_slices.append(np_batch_lbls[b])
            b += 1

        return batch_imgs, np_batch_lbls, target_labels

    def _create_train_batch_item(self, slice_area_spec, is_positive, batch_item_nr):
        """

        :param slice_area_spec: is a tuple (1) slice number (2) target area described by box-four specification
                                i.e. [x.start, y.start, x.stop, y.stop]

        :param is_positive: boolean, indicating a slice which contains at least one positive target voxel
        :param batch_item_nr: the current batch item number we're processing
        :return: (1) input_channels_patch, torch.FloatTensor of shape [3, patch_size, patch_size]
                 (2) lbl_slice: Numpy binary array containing the target pixels [patch_size, patch_size].
                                we only need this array to compute the target roi labels in _generate_train_labels
                 (3) target_label: Binary value indicating whether the slice is positive or negative. Not used
        """
        # first we sample a pixel from the area of interest
        slice_num = slice_area_spec[0]
        input_channels = self.data_set.train_images[slice_num]
        label = self.data_set.train_labels[slice_num]
        slice_x, slice_y = slice_area_spec[1]  # is 1x4 np.array
        # input_channels has size [num_channels, w, h]
        input_channels_patch = input_channels[:, slice_x, slice_y]
        lbl_slice = label[slice_x, slice_y]
        target_label = 1 if np.count_nonzero(lbl_slice) != 0 else 0
        if is_positive and target_label == 0:   # and self.verbose:
            print(("WARNING - Should be positives but no labels present in slice {}".format(slice_num)))
            print(("WARNING ---> slice num {} ({}) contains labels {}".format(slice_num, batch_item_nr,
                                                                                     target_label)))
            print((slice_x, slice_y))
            print((lbl_slice.shape))

        if self.keep_bounding_boxes:
            roi_box_of_four = BoundingBox.convert_slices_to_box_four(slice_x, slice_y)
            self.batch_bounding_boxes[batch_item_nr] = roi_box_of_four

        return torch.FloatTensor(torch.from_numpy(input_channels_patch).float()), lbl_slice, target_label

    def _generate_batch_labels(self, lbl_slice, target_labels, batch_item_nr, is_positive,
                               target_labels_stats_per_roi=None):
        """
        we already calculated the label for the entire patch (assumed to be square).
        Here, we compute the labels for the grid that is produced after maxPool 2 and maxPool 3
        E.g.: patch_size 72x72, has a grids of 2x2 (after maxPool 1) and 4x4 after maxPool 2

        :param lbl_slice:
        :param target_labels: dictionary with keys grid spacing. Currently 1, 4 and 8
        :param batch_item_nr: sequence number for batch item (ID)
        :param is_positive:
        :param target_labels_stats_per_roi: dictionary with keys grid spacing. Values store num of positive voxels
                                            per grid-block (used for evaluation purposes)
        :return: target_labels:
                 A dictionary of torch.LongTensors. Dict keys are grid spacings after maxPooling operations.
                 i.e.: 4 = grid spacing after maxPool 2, 8 = after maxPool 3
                 Each key contains torch tensor of shape [batch_size, grid_size, grid_size]
                target_labels_stats_per_roi if not None
        """

        w, h = lbl_slice.shape
        if is_positive == 0:
            print(("WARNING ---> batch_item_nr {}".format(batch_item_nr)))
            print("")
            raise ValueError("ERROR - _generate_train_labels. is_positive must be equal to True/1, not {}"
                             "".format(is_positive))
        # print("_generate_batch_labels ", w, h, self.num_of_max_pool_layers)
        if is_positive:
            all_labels = {}
            # First split label slice vertically
            for i in np.arange(2, self.num_of_max_pool_layers + 1):
                grid_spacing = int(2**i)
                # omit the [0, ...] at the front of the array, we don't want to split there
                grid_spacings_w = np.arange(0, w, grid_spacing)[1:]
                v_blocks = np.vsplit(lbl_slice, grid_spacings_w)
                all_labels[grid_spacing] = []
                # Second split label slice horizontally
                for block in v_blocks:
                    grid_spacings_h = np.arange(0, h, grid_spacing)[1:]
                    h_blocks = np.hsplit(block, grid_spacings_h)
                    all_labels[grid_spacing].extend(h_blocks)
                # print("_generate_batch_labels ", grid_spacing, len(all_labels[grid_spacing]))
            for grid_spacing, label_patches in all_labels.items():
                grid_target_labels = np.zeros(len(label_patches))
                grid_umap_mask = np.zeros(len(label_patches))
                # REMEMBER: label_patches is a list of e.g. 81 in case of maxPool 3 layer which has
                # a final feature map size of 9x9.
                for i, label_patch in enumerate(label_patches):
                    label_patch = np.array(label_patch)
                    num_of_positives = np.count_nonzero(label_patch)
                    if target_labels_stats_per_roi is not None:
                        target_labels_stats_per_roi[grid_spacing][batch_item_nr][i] = \
                            num_of_positives if num_of_positives != 0 else 0
                    grid_target_labels[i] = 1 if 0 != num_of_positives else 0

                target_labels[grid_spacing][batch_item_nr] = \
                    torch.LongTensor(torch.from_numpy(grid_target_labels).long())
                # torch.LongTensor(torch.from_numpy(grid_target_labels).long())

        if target_labels_stats_per_roi is not None:
            # During testing we're returning the dict (target_labels) and the original target_labels_stats_per_roi
            # which contains per grid_spacing the number of positive values (see method header)
            return target_labels, target_labels_stats_per_roi
        else:
            return target_labels

    def _ablate_input(self, input_channels, target_labels):
        """

        :param input_channels: original e- or u-map of shape [w, h] patch size
        :param target_labels: target rois, binary map, ones indicating voxels to be detected
        :return: altered/enhanced umap around voxels to be detected
        """
        # set voxels in umap to highest uncertainty value (we use normalized u-values between [0, 1]
        input_channels[1][target_labels == 1] = 0.9

        return input_channels

    def _generate_test_batch(self, batch_size=8, keep_batch=False, disrupt_chnnl=None, location_only="ALL"):
        """

        :param batch_size:
        :param keep_batch:
        :param disrupt_chnnl:
        :param location_only: M=only middle slices; AB=only BASE/APEX slices, otherwise ALL slices
        :return:
        """
        if self.last_test_list_idx is None:
            self.last_test_list_idx = 0
        else:
            # we need to increase the index by one to start with the "next" slice from the test set
            self.last_test_list_idx += 1
        self.num_of_skipped_slices = 0
        self.batch_dta_slice_ids = []
        self.batch_pred_probs = []
        self.batch_pred_labels = []
        # The original image size in the dataset
        self.batch_org_image_size = []
        # The width, height of the cropped patch that we use during training/testing of the region detector
        self.batch_patch_size = []
        self.batch_gt_labels = []
        self.batch_patient_ids = []
        self.batch_slice_areas = []
        # stores 1=apex/base or 0=middle slice
        self.batch_extra_labels = []
        self.keep_batch_images = []
        self.keep_batch_target_labels = []
        self.keep_batch_label_slices = []
        self.keep_batch_target_counts = []
        self.target_labels_stats_per_roi = {}
        self.batch_patient_pred_probs = {}
        self.batch_patient_gt_labels = {}
        self.batch_slice_recon_map = {}
        # statistics
        num_of_slices = 0
        num_of_slices_with_errors = 0
        num_of_ba_slices_with_errors = 0
        num_of_ba_slices = 0

        for i in np.arange(2, self.num_of_max_pool_layers + 1):
            grid_spacing = int(2 ** i)
            self.target_labels_stats_per_roi[grid_spacing] = []
        if disrupt_chnnl is not None:
            print(("WARNING - Disrupting input channel {}".format(disrupt_chnnl)))
        # we loop over the slices in the test set, starting
        for list_idx in np.arange(self.last_test_list_idx, self.last_test_list_idx + batch_size):
            target_labels = {}
            target_labels_stats_per_roi = {}

            label = self.data_set.test_labels[list_idx]
            base_apex_slice = self.data_set.test_labels_extra[list_idx]
            roi_area_spec = self.data_set.test_pred_lbl_rois[list_idx]
            slice_x, slice_y = BoundingBox.convert_to_slices(roi_area_spec)

            # print(self.data_set.test_patient_slice_id[list_idx])
            # if self.data_set.test_patient_slice_id[list_idx][0] == "patient085":
            #    pass
            if slice_x.start == 0 and slice_x.stop == 0:
                # IMPORTANT: WE SKIP SLICES THAT DO NOT CONTAIN ANY AUTOMATIC SEGMENTATIONS!
                # does not contain any automatic segmentation mask, continue
                # TODO: 13-03-2019 we changed this. Before we skipped these
                slice_x, slice_y = self._get_center_roi(label)
                # self.num_of_skipped_slices += 1
                # continue
            if location_only == "AB" and not base_apex_slice:
                # ONLY PROCESS BASE/APEX SLICES
                self.num_of_skipped_slices += 1
                continue
            elif location_only == "M" and base_apex_slice:
                # SKIP BASE or APEX slices
                self.num_of_skipped_slices += 1
                continue
            else:
                # we test ALL SLICES
                pass
            # For ablation purposes we can disrupt specific input channels (0-2)
            if disrupt_chnnl is not None:
                input_channels = copy.deepcopy(self.data_set.test_images[list_idx])
                # input image should have 3 channels [3, w, h] mri, uncertainty, auto seg-mask
                # if dataset uses 3d input, then shape is [3, depth, w, h]
                # TODO: not taking into account 3d input at the moment!!!
                if disrupt_chnnl <= 3:
                    _, w, h = input_channels.shape
                    input_channels[disrupt_chnnl] = np.random.randn(w, h)
            else:
                input_channels = self.data_set.test_images[list_idx]
            # test_patient_slice_id contains 4-tuple (patient_id, slice_id, cardiac_phase, frame_id)
            patient_id, slice_id, _, _ = self.data_set.test_patient_slice_id[list_idx]
            self.batch_slice_areas.append(tuple((slice_x, slice_y)))
            self.batch_extra_labels.append(base_apex_slice)
            self.batch_patient_slice_id.append(self.data_set.test_patient_slice_id[list_idx])

            # now slice input and label according to roi specifications (automatic segmentation mask roi)
            # first keep the original shape of the input, we need this later
            # input_channels has size [num_channels, w, h]
            _, w_org, h_org = input_channels.shape
            input_channels_patch = input_channels[:, slice_x, slice_y]
            _, w, h = input_channels_patch.shape

            self.batch_org_image_size.append(tuple((w_org, h_org)))
            self.batch_patch_size.append(tuple((w, h)))
            lbl_slice = label[slice_x, slice_y]
            # does the patch contain any target voxels?
            contains_pos_voxels = 1 if np.count_nonzero(lbl_slice) != 0 else 0
            num_of_slices_with_errors += contains_pos_voxels
            if base_apex_slice:
                if contains_pos_voxels:
                    num_of_ba_slices_with_errors += 1
                num_of_ba_slices += 1

            # Ablation study. Altering uncertainty maps if disrupt_chnnl == 4
            # we only do this for patches that actually contain target seg-errors to be detected.
            if contains_pos_voxels and disrupt_chnnl is not None and disrupt_chnnl == 4:
                input_channels_patch = self._ablate_input(input_channels_patch, lbl_slice)

            num_of_slices += 1
            # store the overall indication whether our slice contains any voxels to be inspected
            target_labels[1] = np.array([contains_pos_voxels])
            target_labels_stats_per_roi[1] = np.array([contains_pos_voxels])
            # construct PyTorch tensor and add a dummy batch dimension in front

            input_channels_patch = torch.FloatTensor(torch.from_numpy(input_channels_patch[np.newaxis]).float())
            self.batch_dta_slice_ids.append(list_idx)
            # initialize dictionary target_labels with (currently) three keys for grid-spacing 1 (overall), 4, 8
            for i in np.arange(2, self.num_of_max_pool_layers + 1):
                grid_spacing = int(2 ** i)
                g_w = int(w / grid_spacing)
                g_h = int(h / grid_spacing)
                # print("INFO - spacing {} - grid size {}x{}={}".format(grid_spacing, g_w, g_h, g_w * g_h))
                # looks kind of awkward, but we always use a batch size of 1
                target_labels[grid_spacing] = torch.zeros((1, g_w * g_h), dtype=torch.long)
                # we need the dummy batch dimension (which is always 1) for the _generate_batch_labels method
                target_labels_stats_per_roi[grid_spacing] = np.zeros((1, g_w * g_h))

            if contains_pos_voxels:
                target_labels, target_labels_stats_per_roi = self._generate_batch_labels(lbl_slice, target_labels,
                                                                                         batch_item_nr=0,
                                                                                         is_positive=contains_pos_voxels,
                                                                                         target_labels_stats_per_roi=
                                                                                         target_labels_stats_per_roi)
            else:
                # no target voxels to inspect in ROI
                pass

            # if on GPU
            if self.cuda:
                input_channels_patch = input_channels_patch.cuda()
                for g_spacing, target_lbls in target_labels.items():
                    # we don't use grid-spacing key "1" (overall binary indication whether patch contains target
                    # voxels (pos/neg example)
                    if g_spacing != 1:
                        target_labels[g_spacing] = target_lbls.cuda()
            # we keep the batch details in lists. Actually only used during debugging to make sure the patches
            # are indeed what we expect them to be.
            if keep_batch:
                self.keep_batch_images.append(input_channels_patch.detach().cpu().numpy())
                self.keep_batch_target_labels.append(target_labels)
                self.keep_batch_label_slices.append(lbl_slice)
                self.keep_batch_target_counts.append(target_labels_stats_per_roi[config_detector.max_grid_spacing])
            for grid_sp in list(target_labels_stats_per_roi.keys()):
                if grid_sp != 1:
                    self.target_labels_stats_per_roi[grid_sp].extend(target_labels_stats_per_roi[grid_sp].flatten())
            # self.input_channels is None, 0=only mri, 1=uncertainty+mri or 2=seg-mask+mri
            if self.input_channels is not None:
                if self.input_channels == 0:
                    input_channels_patch = input_channels_patch[:, self.input_channels].unsqueeze(1)
                else:
                    input_channels_patch = torch.cat((input_channels_patch[:, 0].unsqueeze(1),
                                                      input_channels_patch[:, self.input_channels].unsqueeze(1)),
                                                      dim=1)
            yield input_channels_patch, target_labels
            self.last_test_list_idx = list_idx

        self.batch_size = len(self.batch_dta_slice_ids)

    def visualize_batch(self, grid_spacing=8, index_range=None, base_apex_only=False, sr_threshold=0.5, patient_id=None,
                        right_column="map", left_column="error_roi", data_handler=None, do_save=False,
                        heat_map_handler=None, alpha=0.5, model_handler=None):
        """

        :param grid_spacing:
        :param index_range:
        :param base_apex_only:
        :param sr_threshold:
        :param patient_id:
        :param left_column: map=uncertainty map; seg_error=unfiltered segmentation errors; auto=automatic seg-mask
                            error_roi=filtered seg-errors (supervision)
        :param right_column: error_roi=filtered seg-errors (supervision);
        :param data_handler
        :return:
        """
        if self.trans_dict is None:
            self.fill_trans_dict()

        mycmap = transparent_cmap(plt.get_cmap('jet'), alpha=0.2)
        my_seg_mag_trans = SegColorMap(alpha=alpha)
        error_cmap = transparent_cmap(plt.get_cmap('jet'), alpha=alpha)
        my_cmap_umap = transparent_cmap(plt.get_cmap('plasma'), alpha=alpha)
        
        if index_range is None:
            if patient_id is None:
                raise ValueError("ERROR - index_range or patient_id needs to be specified!")
            index_range = [0, self.batch_size]
        slice_idx_generator = np.arange(index_range[0], index_range[1])
        if patient_id is not None:
            number_of_slices = 40
        else:
            number_of_slices = slice_idx_generator.shape[0]

        width = 16
        height = number_of_slices * 8
        print(("number_of_slices {} height, width {}, {}".format(number_of_slices, height, width)))
        columns = 4
        rows = number_of_slices * 2  # 2 because we do double row plotting
        row = 0
        fig = plt.figure(figsize=(width, height))
        heat_map = None
        for idx in slice_idx_generator:
            slice_num = self.batch_dta_slice_ids[idx]
            base_apex_slice = self.batch_extra_labels[idx]
            # pat_slice_id: contains tuple (patient_id, slice_id (increased + 1), 0=ES/1=ED)
            # NOTE: for new DataHandlerPhase the last entry is equal to frame_id instead of 0/1 for ES/ED
            pat_slice_id = self.batch_patient_slice_id[idx]
            p_id = pat_slice_id[0]
            frame_id = pat_slice_id[2]
            if patient_id is not None and p_id != patient_id:
                # We only want to visualize slices of a specific patient
                continue
            if self.keep_batch_label_slices is not None and len(self.keep_batch_label_slices) != 0:
                target_slice = self.keep_batch_label_slices[idx]
            else:
                print("WARNING - self.batch_label_slices is empty")
            if not self.is_train:
                # during testing batch_images is a list with numpy arrays of shape [1, 3, w, h]
                # during training it's soley a numpy array of shape [batch_size, 3, w, h]
                # Hence, during testing we need to get rid of the first dummy batch dimension
                # idx = list item, first 0 = dummy batch dimension always equal to 1, second 0 = image channel
                image_slice = self.keep_batch_images[idx][0][0].detach().cpu().numpy()
                uncertainty_slice = self.keep_batch_images[idx][0][1].detach().cpu().numpy()
                # first list key/index and then dict key 1 for overall indication
                target_lbl_binary = self.keep_batch_target_labels[idx][1][0]
                target_lbl_binary_grid = self.keep_batch_target_labels[idx][grid_spacing][0]
                if len(self.batch_pred_probs) != 0:
                    # the model softmax predictions are store in batch property (list) batch_pred_probs if we enabled
                    # keep_batch during testing. The shape of np array is [1, w/grid_spacing, h/grid_spacing]
                    # and we need to get rid of first batch dimension.
                    p = self.batch_pred_probs[idx]
                    pred_probs = np.squeeze(p)
                    w, h = image_slice.shape
                    heat_map, grid_map, target_lbl_grid = create_grid_heat_map(pred_probs, grid_spacing, w, h,
                                                                               prob_threshold=sr_threshold)
            else:
                # training
                image_slice = self.keep_batch_images[idx][0].detach().cpu().numpy()
                uncertainty_slice = self.keep_batch_images[idx][1].detach().cpu().numpy()
                target_lbl_binary = self.keep_batch_target_labels[idx][1].detach().cpu().numpy()
                target_lbl_binary_grid = self.keep_batch_target_labels[idx][grid_spacing].data.cpu().numpy()
            lbl_slice = self.keep_batch_label_slices[idx]
            w, h = image_slice.shape
            if base_apex_only and not base_apex_slice:
                continue
            ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            ax1.set_title("Dataset slice {} ({}) (base/apex={})".format(slice_num, idx + 1, base_apex_slice))
            ax1.imshow(image_slice, cmap=cm.gray)
            ax1.set_xticks(np.arange(-.5, h, grid_spacing))
            ax1.set_yticks(np.arange(-.5, w, grid_spacing))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            plt.grid(True)

            ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
            ax2.imshow(image_slice, cmap=cm.gray)
            cardiac_phase = pat_slice_id[2]
            patient_slice_id = pat_slice_id[1] - 1
            slice_x, slice_y = self.batch_slice_areas[idx]
            if left_column == "seg_error":
                # the segmentation errors for this slice. Note, when training with raw seg-errors this is equal to
                seg_error_volume = data_handler.get_pred_labels_errors(p_id, cardiac_phase, mc_dropout=False)
                seg_error_slice = seg_error_volume[..., patient_slice_id]
                multi_auto_seg_error = convert_to_multiclass(seg_error_slice)
                multi_auto_seg_error = my_seg_mag_trans.convert_multi_labels(multi_auto_seg_error)
                multi_auto_seg_error = multi_auto_seg_error[slice_x, slice_y]
                ax1.imshow(multi_auto_seg_error, cmap=my_seg_mag_trans.cmap)
                # seg_error_slice = (np.sum(seg_error_slice, axis=0) != 0).astype(np.int)
                # seg_error_slice = seg_error_slice[slice_x, slice_y]
                # ax1.imshow(multi_auto_seg_error, cmap=error_cmap)
                # ax1.imshow(uncertainty_slice, cmap=mycmap)
            elif left_column == "map":
                ax1.imshow(uncertainty_slice, cmap=my_cmap_umap)
            elif left_column == "error_roi":
                # blurred_lbl_slice = gaussian_filter(lbl_slice.astype(np.float), sigma=2)
                ax1.imshow(lbl_slice, cmap=error_cmap)
            elif left_column == "auto":
                # automatic seg-mask
                multi_auto_mask = self.keep_batch_images[idx][0][2].detach().cpu().numpy()
                # multi_auto_mask has already multi-labels which we'll translate into different color codings
                multi_auto_mask = my_seg_mag_trans.convert_multi_labels(multi_auto_mask)
                ax1.imshow(multi_auto_mask, cmap=my_seg_mag_trans.cmap)
            elif left_column == "recon":
                voxel_heat_map = heat_map_handler.get_heat_map(p_id, cardiac_phase)
                voxel_heat_map = voxel_heat_map[slice_x, slice_y, patient_slice_id]
                voxel_heat_map[voxel_heat_map < sr_threshold] = 0.
                voxel_heat_map_plot = ax1.imshow(voxel_heat_map, cmap=mycmap, vmin=0., vmax=1.)
                ax1.set_aspect('auto')
                fig.colorbar(voxel_heat_map_plot, ax=ax1, fraction=0.046, pad=0.04)

            ax2.set_title("Contains ta-voxels {} {}".format(int(target_lbl_binary), pat_slice_id))
            fontsize = 10 if grid_spacing == 4 else 15
            # target rois (supervision for detection task
            if right_column == "error_roi":
                ax2.imshow(lbl_slice, cmap=error_cmap)
            elif right_column == "seg_error":
                # the segmentation errors for this slice. Note, when training with raw seg-errors this is equal to
                seg_error_volume = data_handler.get_pred_labels_errors(p_id, cardiac_phase, mc_dropout=False)
                seg_error_slice = seg_error_volume[..., patient_slice_id]
                seg_error_slice = (np.sum(seg_error_slice, axis=0) != 0).astype(np.int)
                seg_error_slice = seg_error_slice[slice_x, slice_y]
                ax2.imshow(seg_error_slice, cmap=error_cmap)
            else:
                pass
            if heat_map is not None:
                # target_slice: binary indication whether the tile needs to be detected or not
                ax2.imshow(target_slice, cmap=mycmap)
                heat_map_plot = ax2.imshow(heat_map, cmap=mycmap, vmin=0, vmax=1)
                for i, map_index in enumerate(zip(grid_map[0], grid_map[1])):
                    z_i = target_lbl_binary_grid[i]
                    if z_i == 1:
                        # BECAUSE, we are using ax2 with imshow, we need to swap x, y coordinates of map_index
                        ax2.text(map_index[1], map_index[0], '{}'.format(z_i), ha='center', va='center', fontsize=25,
                                 color="y")
                ax2.set_aspect('auto')
                fig.colorbar(heat_map_plot, ax=ax2, fraction=0.046, pad=0.04)
            else:
                _, grid_map, _ = create_grid_heat_map(None, grid_spacing, w, h,
                                                      prob_threshold=0.5)
                for i, map_index in enumerate(zip(grid_map[0], grid_map[1])):
                    z_i = target_lbl_binary_grid[i]
                    if z_i == 1:
                        # BECAUSE, we are using ax2 with imshow, we need to swap x, y coordinates of map_index
                        ax2.text(map_index[1], map_index[0], '{}'.format(z_i), ha='center', va='center',
                                 fontsize=fontsize, color="b")

            ax2.set_xticks(np.arange(-.5, h, grid_spacing))
            ax2.set_yticks(np.arange(-.5, w, grid_spacing))
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            plt.grid(True)
            # plt.axis("off")
            row += 2

        fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
        if do_save:
            # TODO path is NOT CORRECT!!!
            fold_id = model_handler.fold_id

            abs_path_fold = os.path.join(self.app_config.data_dir, "fold" + str(fold_id))
            fig_path = os.path.join(abs_path_fold, "figures")

            if not os.path.isdir(fig_path):
                os.makedirs(fig_path)
            fig_path = os.path.join(fig_path, model_handler.model_name)
            if not os.path.isdir(fig_path):
                os.makedirs(fig_path)
            fig_path = os.path.join(fig_path, model_handler.type_of_map)
            if not os.path.isdir(fig_path):
                os.makedirs(fig_path)
            fig_name = patient_id + "_" + "regions"
            fig_name = os.path.join(fig_path, fig_name + ".pdf")

            plt.savefig(fig_name, bbox_inches='tight')
            print(("INFO - Successfully saved fig %s" % fig_name))
        plt.show()

