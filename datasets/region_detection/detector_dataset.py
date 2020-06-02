import numpy as np
from collections import OrderedDict
from networks.detection.general_setup import config_detector

from utils.box_utils import BoundingBox, find_bbox_object
from utils.box_utils import adjust_roi_bounding_box


class RegionDetectorDataSet(object):

    pixel_dta_type = 'float32'
    pad_size = 0
    num_of_augs = 4

    def __init__(self, num_of_channels=3, mc_dropout=False):
        """

        :param num_of_channels: 2 or 3
        :param mc_dropout:

        """

        self.num_of_channels = num_of_channels
        # mc_dropout
        self.mc_dropout = mc_dropout
        # The key of the dictionaries is patient_id.
        self.train_images = []
        self.train_labels = []
        self.train_labels_extra = []
        # list of numpy [N, 4] arrays that describe the target tissue area of the automatic predictions
        # we use these to sample from when generating batches (in order to stay close to the target tissue area)
        self.train_pred_lbl_rois = []
        self.train_lbl_rois = []
        self.train_extra_labels = []
        # contain numpy array with shape [#slices, 3, w, h]
        self.test_images = []
        # contain numpy array with shape [#slices]
        self.test_labels = []
        # indication apex/base
        self.test_labels_extra = []
        self.test_pred_lbl_rois = []
        # store box coordinates (four) for the target roi labels (can be multiple). We don't use these currently
        self.test_lbl_rois = []
        # in case we use the RegionDataSet for testing only, we sometimes want to have the original
        # segmentation references
        # in order to investigate the errors the model makes. Hence, we can store them in this list
        # IMPORTANT: they are not augmented but in their original shape. We store them per patient. So each
        # volume has shape [2, w, h, #slices]
        self.test_ref_segmentation = {}
        # stores the padding we applied to the test images. tuple of tuples e.g. ((2, 1), (5, 3))
        self.test_paddings = []
        # contain numpy array with shape [#slices, 4] 1) phase 2) slice id 3) mean-dice/slice 4) patient id
        self.test_extra_labels = []
        # actually "size" here is relate to number of patients
        self.size_train = 0
        self.size_test = 0
        self.train_patient_ids = []
        self.train_patient_slice_id = []
        self.train_patient_num_of_slices = {}
        self.test_patient_ids = []
        # dict with number of slices per patient. NOTE! This is for ONE cardiac phase e.g. 8 than 16 for ES+ED
        self.test_patient_num_of_slices = {}
        self.test_patient_slice_id = []
        # translation dictionary from patient id to list index numbers. Key=patient_id,
        # value=tuple(is_train, [<indices>])
        self.trans_dict = OrderedDict()
        self.img_spacings = {}
        # ROI statistics, indices meaning: 0=approx # of target grids
        #                                  1=total ROIs, 2=ROIS in base/apex slices, 3=ROIS in other slices
        self.roi_stats = {"train": np.zeros(4).astype(np.int), "test": np.zeros(4).astype(np.int)}
        self.config = config_detector

    def get_patient_ids(self, is_train=True):
        if is_train:
            return self.train_patient_ids
        else:
            return self.test_patient_ids

    def get_size(self, is_train=True):
        if is_train:
            return self.size_train
        else:
            return self.size_test

    def load_from_file(self):
        raise NotImplementedError()

    def collect_roi_data(self):
        roi_areas = []
        roi_aspect_ratio = []

        def loop_over_rois(roi_area_list):
            for b in np.arange(0, len(roi_area_list), 4):
                roi_array = roi_area_list[b]
                # roi_array is Nx4 np array
                for i in np.arange(roi_array.shape[0]):
                    bbox = BoundingBox.create(roi_array[i])
                    roi_areas.append(bbox.area)
                    roi_aspect_ratio.append(bbox.width / float(bbox.height))

        loop_over_rois(self.train_lbl_rois)
        loop_over_rois(self.test_lbl_rois)
        roi_areas = np.array(roi_areas)
        roi_aspect_ratio = np.array(roi_aspect_ratio)
        return roi_areas, roi_aspect_ratio

    def augment_data(self, input_chnl1, input_chnl2, input_chnl3, label_slices, is_train=False,
                     do_rotate=False, patient_id=None, cardiac_phase=None, frame_id=None, pred_labels_vol=None):
        """
        Augments image slices by rotating z-axis slices for 90, 180 and 270 degrees

        :param input_chnl1: [w, h, #slices] original mri
        :param input_chnl2: [w, h, #slices] uncertainty maps
        :param input_chnl3: [w, h, #slices] automatic reference
        :param label_slices: [w, h, #slices] binary values indicating whether voxel should be corrected
        :param is_train: boolean
        :param do_rotate: boolean, if True (currently only for training images) then each slice is rotated
                          three times (90, ..., 270)
        :param patient_id
        :param cardiac_phase
        :param pred_labels_vol: [w, h, #slices] contains the automatic segmentation mask
                                if we use soft labels, then input_chnl3 contains predicted probabilities and not the
                                hard labels. But we need hard labels to determine the cropped ROI for the input

        :return None (fills train/test arrays of object)

        """

        def rotate_slice(mri_img_slice, uncertainty_slice, pred_lbl_slice, label_slice, is_train=False,
                         pat_slice_id=None, base_apex_slice=None, pred_labels_slice=None, patient_id=None):
            """

            :param mri_img_slice: [w, h] original mri image
            :param uncertainty_slice  [w, h] uncertainty map
            :param pred_lbl_slice: [w, h] predicted seg mask
            :param label_slice: [w, h]

            :param is_train: boolean
            :return: None
            """

            for rots in range(RegionDetectorDataSet.num_of_augs):

                self.add_image_to_set(is_train, mri_img_slice, uncertainty_slice, pred_lbl_slice, label_slice,
                                      pat_slice_id=pat_slice_id, base_apex_slice=base_apex_slice,
                                      pred_labels_slice=pred_labels_slice, patient_id=patient_id)
                # rotate for next iteration
                if mri_img_slice.ndim == 2:
                    rot_axis = (0, 1)
                else:
                    rot_axis = (1, 2)
                mri_img_slice = np.rot90(mri_img_slice, axes=rot_axis)
                uncertainty_slice = np.rot90(uncertainty_slice, axes=rot_axis)
                pred_lbl_slice = np.rot90(pred_lbl_slice, axes=rot_axis)
                pred_labels_slice = np.rot90(pred_labels_slice)
                label_slice = np.rot90(label_slice)

        w, h, num_of_slices = input_chnl1.shape
        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(num_of_slices):
            pat_slice_id = tuple((patient_id, z+1, cardiac_phase, frame_id))
            input_chnl1_slice = input_chnl1[:, :, z]
            input_chnl2_slice = input_chnl2[:, :, z]
            input_chnl3_slice = input_chnl3[:, :, z]
            label_slice = label_slices[:, :, z]
            pred_labels_slice = pred_labels_vol[:, :, z]
            w, h = label_slice.shape
            # NOTE: this is only an approximation of the number of target grids for which we predict the presence of
            # voxels to be inspected. We currently tile the images in grids with spacing 8 voxels (see config_detector)
            num_of_grids = int((w / 8) * (h / 8))
            if 0 != np.count_nonzero(label_slice):
                # label_bbox contains np array [N, 4], coordinates of ROI area around target region we're interested
                # in. label_slice_filtered: filtered by 2D 4-connected structure.
                # Meaning, that potential target voxels that are not part of larger 4-connected components will be
                # discarded. Please also see config_detector.min_roi_area, currently = 2!
                # label_bbox, label_slice_filtered, roi_bbox_area = find_multiple_connected_rois(label_slice, padding=1)
                # 16-11-2018: Changed this. We're not anymore filtering target rois based on connected components
                # 09-01-2019: filter target rois max. Target voxels must be connected in a star structure
                #             resulted in slightly worse results compared to no filtering
                # label_slice = filter_single_voxels(label_slice)
                self.roi_stats["train" if is_train else "test"][0] += num_of_grids
                # increase total #ROIS
                # 16-11-2018 changed this to number of voxels
                # num_of_rois = label_bbox.shape[0]
                num_of_voxel_rois = np.count_nonzero(label_slice)
                self.roi_stats["train" if is_train else "test"][1] += num_of_voxel_rois
                if z == 0 or z == (num_of_slices - 1):
                    self.roi_stats["train" if is_train else "test"][2] += num_of_voxel_rois
                    base_apex_slice = 1
                else:
                    self.roi_stats["train" if is_train else "test"][3] += num_of_voxel_rois
                    base_apex_slice = 0

            else:
                # determine base/apex or middle slice
                if z == 0 or z == (num_of_slices - 1):
                    base_apex_slice = 1
                else:
                    base_apex_slice = 0
                # 16-11-2018 disabled this because we're not anymore filtering the target rois on connected comps
                # label_slice_filtered = label_slice
            if do_rotate:
                rotate_slice(input_chnl1_slice, input_chnl2_slice, input_chnl3_slice, label_slice,
                             is_train, pat_slice_id=pat_slice_id, base_apex_slice=base_apex_slice,
                             pred_labels_slice=pred_labels_slice, patient_id=patient_id)
            else:
                # Note: for the TEST set we also use the label_slice object.
                self.add_image_to_set(is_train, input_chnl1_slice, input_chnl2_slice, input_chnl3_slice,
                                      label_slice, pat_slice_id, base_apex_slice,
                                      pred_labels_slice, patient_id=patient_id)

    def add_image_to_set(self, is_train, input_chnl1_slice, input_chnl2_slice, input_chnl3_slice, label_slice,
                         pat_slice_id=None, base_apex_slice=None, pred_labels_slice=None, patient_id=None):
        if is_train:
            p_slice1 = np.pad(input_chnl1_slice, RegionDetectorDataSet.pad_size, 'constant',
                              constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
            p_slice2 = np.pad(input_chnl2_slice, RegionDetectorDataSet.pad_size, 'constant',
                              constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
            p_slice3 = np.pad(input_chnl3_slice, RegionDetectorDataSet.pad_size, 'constant',
                              constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
            # we get the bounding box for the predicted aka automatic segmentation mask. we use this for batch
            # generation
            pred_lbl_roi = find_bbox_object(pred_labels_slice, padding=0)
            # should result again in [3, w+pad_size, h+pad_size]
            # print(pat_slice_id, p_slice1.shape, p_slice2.shape, p_slice3.shape)
            padded_input_slices = np.concatenate((p_slice1[np.newaxis], p_slice2[np.newaxis], p_slice3[np.newaxis]))
        else:
            # find_bbox_object returns BoundingBox object. We're looking for bounding boxes of the automatic
            # segmentation mask. If not None (automatic seg mask) then check the size of the bbox. During
            # validation & testing we're only processing mri slices with an automatic segmentation mask.
            # Because we've prior knowlegde about the dataset, we know that the mask will be always smaller than
            # the original image, hence, we NEVER have to pad the image in order to make sure that the size (w, h)
            # is dividable by max_grid_spacing (currently 8).
            pred_lbl_roi = find_bbox_object(pred_labels_slice, padding=2)

            # in case we are dealing with slices that do not contain an automatic seg-mask we don't need to find
            # the ROI. We will even not use these slices during testing because we're only interested in slices with
            # an automatic seg-mask TODO: 13-03-2019 we changed this. currently training with these! See batch-handler
            # Is slice_idx=len(self.test_images) then procedure will print slices bigger than
            # training patch_size. We use this for debugging purposes.
            if not pred_lbl_roi.empty:
                # if the slice contains an automatic segmentation mask, we determine the ROI around it. We use the
                # ROI during training in order to "crop" the image to the area we're actually considering for
                # detection of the regions we want to inspect.
                if is_train:
                    patch_size = self.config.patch_size
                else:
                    patch_size = self.config.test_patch_size
                w, h = input_chnl1_slice.shape
                pred_lbl_roi = adjust_roi_bounding_box(pred_lbl_roi, w, h, slice_idx=None, patch_size=patch_size,
                                                       max_grid_spacing=self.config.max_grid_spacing)

            padded_input_slices = np.concatenate((input_chnl1_slice[np.newaxis], input_chnl2_slice[np.newaxis],
                                                  input_chnl3_slice[np.newaxis]))
        # we get the bounding boxes for the different target rois in box_four format (x.start, y.start, ...)
        # we use these when generating the batches, because we want to make sure that for the positive batch items
        # at least ONE target rois is in the FOV i.e. included in the patch that we sample from a slice
        # 16-11-2018 we're not anymore identifying individual rois for the target regions, but use one bounding box
        # for the complete target region. We do this because we only exploit the region to create batches.
        # bbox_for_rois = find_box_four_rois(label_slice)

        roi_bbox_area = find_bbox_object(label_slice, padding=1)
        if roi_bbox_area.empty:
            roi_bbox_area = np.empty((0, 4))
        else:
            roi_bbox_area = roi_bbox_area.box_four[np.newaxis]

        if is_train:
            self.train_images.append(padded_input_slices)
            self.train_labels.append(label_slice)
            # we add a new axis to np array of shape [4] i.e. [1, 4], because caused by earlier code we expect
            # the first dimension to indicate the number of target rois (when we used individual bounding boxes around
            # the target rois).
            self.train_lbl_rois.append(roi_bbox_area)
            self.trans_dict[patient_id][1].append(len(self.train_images) - 1)
            self.train_pred_lbl_rois.append(pred_lbl_roi.box_four)
            self.train_patient_slice_id.append(pat_slice_id)
            self.train_labels_extra.append(base_apex_slice)
        else:
            self.test_images.append(padded_input_slices)
            self.test_labels.append(label_slice)
            self.test_lbl_rois.append(roi_bbox_area)
            self.trans_dict[patient_id][1].append(len(self.test_images) - 1)
            self.test_pred_lbl_rois.append(pred_lbl_roi.box_four)
            self.test_patient_slice_id.append(pat_slice_id)
            self.test_labels_extra.append(base_apex_slice)

    @staticmethod
    def collapse_roi_maps(target_roi_maps):
        """
        our binary roi maps specify the voxels to "inspect" per tissue class, for training this distinction
        doesn't matter, so here, we collapse the classes to 1
        :param target_roi_maps: [#slices, #classes, w, h]
        :return: [#slices, w, h]

        """
        target_roi_maps = np.sum(target_roi_maps, axis=1)  # sum over classes
        # kind of logical_or over all classes. if one voxels equal to 1 or more, than set voxel to 1 which means
        # we need to correct/inspect that voxel
        target_roi_maps[target_roi_maps >= 1] = 1
        return target_roi_maps.astype(np.int)

    @staticmethod
    def convert_to_multilabel(labels, multiclass_idx):
        """
        Assuming label_slice has shape [#slices, #classes, w, h]

        :param label_slice:
        :param multiclass_idx:  np.array of shape nclasses with values of new labels
        :return:
        """
        num_slices, nclasses, w, h = labels.shape
        multilabel_slice = np.zeros((num_slices, w, h))
        for slice_id in np.arange(num_slices):
            lbl_slice = np.zeros((w, h))
            for cls_idx in np.arange(nclasses):
                if cls_idx != 0:
                    lbl_slice[labels[slice_id, cls_idx] == 1] = multiclass_idx[cls_idx]
            multilabel_slice[slice_id] = lbl_slice
        return multilabel_slice

    @staticmethod
    def remove_padding(image):
        """

        :param image:
        :return:
        """
        if RegionDetectorDataSet.pad_size > 0:
            if image.ndim == 2:
                # assuming [w, h]
                return image[RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size,
                             RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size]
            elif image.ndim == 3:
                # assuming [#channels, w, h]
                return image[:, RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size,
                             RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size]
            elif image.ndim == 4:
                # assuming [#channels, w, h, #slices]
                return image[:, RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size,
                             RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size, :]
        else:
            return image


