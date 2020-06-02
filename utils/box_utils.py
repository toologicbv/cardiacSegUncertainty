import copy
from collections import OrderedDict
from scipy import ndimage
from scipy.ndimage.measurements import label
import numpy as np
import matplotlib.patches as patches


def filter_single_voxels(label_slice):

    structure = [[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]]
    new_label_slice = copy.deepcopy(label_slice)
    cc_labels, n_comps = label(label_slice, structure=structure)
    for i_comp in np.arange(1, n_comps + 1):
        comp_mask = cc_labels == i_comp
        comp_mask_size = np.count_nonzero(comp_mask)
        if comp_mask_size <= 1:
            new_label_slice[cc_labels == i_comp] = 0

    return new_label_slice


def find_multiple_connected_rois(label_slice, padding=1, min_size=2):
    """

    :param label_slice: has shape [w, h] and label values are binary (i.e. no distinction between tissue classes)

    :param padding: in fact extending the roi area that we detected. Because the target structure is of connectivity
                    six (3x3), we at least extend by 1 pixel to both sides
    :param min_size: the minimum area size of the 2D 4-connected component
    :return: roi_binary_mask: the new label_slice containing binary values indicating whether a voxels should
             be detected as "to be examined" voxel
    """
    structure = [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
    # find 4-connected components in slice
    cc_labels, n_comps = label(label_slice, structure=structure)
    roi_boxes = np.empty((0, 4))
    roi_box_areas = []
    roi_binary_mask = np.zeros_like(label_slice)
    total_connected_roi_size = 0
    total_size_omitted = 0
    rois_omitted = 0
    for i_comp in np.arange(1, n_comps + 1):
        comp_mask = cc_labels == i_comp
        roi_slice_x, roi_slice_y = ndimage.find_objects(comp_mask)[0]
        comp_mask_size = np.count_nonzero(comp_mask)
        # roi_slice_x contains [x_low, x_high], roi_slice_y contains [y_low, y_high]
        # roi boxes [Nx4] with x_low, y_low, x_high, y_high
        roi_box = BoundingBox(roi_slice_x, roi_slice_y, padding=padding)
        if comp_mask_size >= min_size:
            roi_box_areas.append(roi_box.area)
            roi_boxes = np.concatenate((roi_boxes, roi_box.box_four[np.newaxis])) if roi_boxes.size else \
                roi_box.box_four[np.newaxis]

            roi_binary_mask[cc_labels == i_comp] = 1
            total_connected_roi_size += comp_mask_size
        else:
            # we discard voxels of the auto-seg mask as targets (for detection) if the 4-connected component
            # comprises less than min_size=currently 2 (see config file) voxels.
            # roi_binary_mask is already set to zero for these voxels
            total_size_omitted += comp_mask_size
            rois_omitted += 1
            pass
    del cc_labels
    del comp_mask

    return roi_boxes, roi_binary_mask, roi_box_areas


def find_bbox_object(multi_label_slice, threshold_pixel_value=0, padding=0):
    # multi_label_slice slice [w, h]. all pixels != 0 belong to the automatic segmentation mask
    # threshold_pixel_value (float): we're trying these bboxes also for the uncertainty maps.
    # but basically all pixels have values above 0. Experimenting with this.
    binary_mask_slice = (multi_label_slice > threshold_pixel_value).astype(np.bool)
    if 0 != np.count_nonzero(binary_mask_slice):
        roi_slice_x, roi_slice_y = ndimage.find_objects(binary_mask_slice == 1)[0]
    else:
        roi_slice_x, roi_slice_y = slice(0, 0, None), slice(0, 0, None)
        padding = 0

    roi_box = BoundingBox(roi_slice_x, roi_slice_y, padding=padding)

    return roi_box


class BoundingBox(object):

    def __init__(self, slice_x, slice_y, padding=0):
        # roi_slice_x contains [x_low, x_high], roi_slice_y contains [y_low, y_high]
        # roi box_four [Nx4] with x_low, y_low, x_high, y_high
        self.empty = False
        slice_x = slice(slice_x.start - padding, slice_x.stop + padding, None)
        slice_y = slice(slice_y.start - padding, slice_y.stop + padding, None)
        self.slice_x = slice_x
        self.slice_y = slice_y
        if slice_x.stop - slice_x.start == 0 or slice_x.start < 0:
            self.empty = True
        self.padding = padding
        self.xy_left = tuple((slice_y.start, slice_x.start))
        # actually we switched height and width because we're
        self.width = slice_x.stop - slice_x.start
        self.height = slice_y.stop - slice_y.start
        self.area = self.height * self.width
        self.box_four = np.array([slice_x.start, slice_y.start,
                                  slice_x.stop, slice_y.stop])
        # create the default rectangular that we can use for plotting (red edges, linewidth=1)
        self.rectangular_patch = self.get_matplotlib_patch()

    def get_matplotlib_patch(self, color='r', linewidth=1):
        rect = patches.Rectangle(self.xy_left, self.height, self.width, linewidth=linewidth, edgecolor=color,
                                 facecolor='none')
        return rect

    @staticmethod
    def create(box_four, padding=0):
        """

        :param box_four: np array of shape [4] with x_low, y_low, x_high, y_high
        :param padding:
        :return: BoundingBox object
        """
        slice_x, slice_y = BoundingBox.convert_to_slices(box_four)
        slice_y = slice(box_four[1], box_four[3], None)
        return BoundingBox(slice_x, slice_y, padding=padding)

    @staticmethod
    def convert_to_slices(box_four):
        box_four = box_four.astype(np.int)
        slice_x = slice(box_four[0], box_four[2], None)
        slice_y = slice(box_four[1], box_four[3], None)
        return slice_x, slice_y

    @staticmethod
    def convert_slices_to_box_four(slice_x, slice_y):
        return np.array([slice_x.start, slice_y.start, slice_x.stop, slice_y.stop])


def find_box_four_rois(label_slice, padding=0):
    """

    :param label_slice: has shape [w, h] and label values are binary (i.e. no distinction between tissue classes)


    :return:
    """
    structure = [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
    cc_labels, n_comps = label(label_slice, structure=structure)
    roi_boxes = np.empty((0, 4))

    for i_comp in np.arange(1, n_comps + 1):
        comp_mask = cc_labels == i_comp
        roi_slice_x, roi_slice_y = ndimage.find_objects(comp_mask)[0]
        roi_box = BoundingBox(roi_slice_x, roi_slice_y, padding=padding)
        roi_boxes = np.concatenate((roi_boxes, roi_box.box_four[np.newaxis])) if roi_boxes.size else \
            roi_box.box_four[np.newaxis]

    return roi_boxes


def adjust_roi_bounding_box(roi_bbox, w_img, h_img, slice_idx=None, patch_size=(80, 80), max_grid_spacing=8):
    """

    :param roi_bbox: BoundingBox object
    :param w_img: width of the original image the patch is extracted from
    :param h_img: height of the original image
            We need w_img and h_img to check whether how we need to adjust the bounding box to fit the image.
            E.g. The patch size is 112^2, and the current bbox has w, h 30, 53 we want the bbox to have size 112^2
            and still contain our target region and fit the original image (otherwise we're in trouble when
            generating batches (train or test).
    :param slice_idx: only for debugging purposes, so we can find back the slice we're processing
    :param patch_size: tuple
    :param max_grid_spacing: integer
    :return: new bounding box around e.g. automatic segmentation mask, that fits the grid spacing we defined
             e.g. 8 voxels.
    """
    # patch_size that is used during training, currently 72x72.
    half_width = patch_size[0] / 2
    half_height = patch_size[1] / 2
    w, h = roi_bbox.width, roi_bbox.height

    tiny_w = False
    tiny_h = False
    # most probably our bbox width is smaller than the patch size we're training testing on. Same for height.
    # so we have to make sure first that the
    if w < patch_size[0]:
        mid_point = int((roi_bbox.slice_x.stop + roi_bbox.slice_x.start) / 2)
        if mid_point - half_width >= 0:
            if mid_point + half_width <= w_img:
                roi_bbox.slice_x = slice(mid_point - half_width, mid_point + half_width, 0)
            else:
                roi_bbox.slice_x = slice(w_img - patch_size[0], w_img, 0)
        else:
            roi_bbox.slice_x = slice(0, patch_size[0], 0)
        roi_bbox.width = roi_bbox.slice_x.stop - roi_bbox.slice_x.start
        w = roi_bbox.width
        tiny_w = True
        # print("New slice_x ", pred_lbl_roi.slice_x, mid_point, pred_lbl_roi.width)
    else:
        if slice_idx is not None:
            print(("WARNING - width >= patch_size - slice {}".format(slice_idx)))
    # same for the height, what we did above for the width
    if h < patch_size[0]:
        mid_point = int((roi_bbox.slice_y.stop + roi_bbox.slice_y.start) / 2)
        if mid_point - half_height >= 0:
            if mid_point + half_height <= h_img:
                roi_bbox.slice_y = slice(mid_point - half_height, mid_point + half_height, 0)
            else:
                roi_bbox.slice_y = slice(h_img - patch_size[1], h_img, 0)
        else:
            roi_bbox.slice_y = slice(0, patch_size[1], 0)
        roi_bbox.height = roi_bbox.slice_y.stop - roi_bbox.slice_y.start
        h = roi_bbox.height
        tiny_h = True
        # print("New slice_y ", pred_lbl_roi.slice_y, mid_point, pred_lbl_roi.height)
    else:
        if slice_idx is not None:
            print(("WARNING - height >= patch_size - slice {}".format(slice_idx)))
    # Finally make sure that the patch fits our grid spacing, hence, a multiple of max_grid_spacing (probably 8 voxels)
    # so that width and height of bbox fit the downsampling path of the network.
    if w % max_grid_spacing != 0:
        extend_w = max_grid_spacing - (w % max_grid_spacing)
    else:
        extend_w = 0
    if h % max_grid_spacing != 0:
        extend_h = max_grid_spacing - (h % max_grid_spacing)
    else:
        extend_h = 0
    # if our mask is greater than the training patch size (currently 72x72) then we make it even bigger
    if not tiny_w:
        extend_w += max_grid_spacing
    if not tiny_h:
        extend_h += max_grid_spacing

    if extend_w % 2 != 0:
        extend_w_x = extend_w / 2
        extend_w_y = extend_w - extend_w_x
    else:
        extend_w_x = extend_w / 2
        extend_w_y = extend_w_x
    if extend_h % 2 != 0:
        extend_h_x = extend_h / 2
        extend_h_y = extend_h - extend_h_x
    else:
        extend_h_x = extend_h / 2
        extend_h_y = extend_h_x

    slice_x = slice(roi_bbox.slice_x.start - extend_w_x, roi_bbox.slice_x.stop + extend_w_y, 0)
    slice_y = slice(roi_bbox.slice_y.start - extend_h_x, roi_bbox.slice_y.stop + extend_h_y, 0)

    new_bouding_box = BoundingBox(slice_x, slice_y)
    # print("Before ", pred_lbl_roi_old.width, pred_lbl_roi_old.height)
    # print("After ", pred_lbl_roi.width, pred_lbl_roi.height, extend_w, extend_h)
    return new_bouding_box


def generate_bbox_target_roi(labels, bg_classes=(0, 4)):
    """

    :param labels: binary labels [#classes, x, y, z]
    :param bg_classes
    :return:
    """
    num_of_slices = labels.shape[3]
    pat_slice_rois = OrderedDict()
    num_of_classes = int(labels.shape[0])
    for slice_id in np.arange(num_of_slices):
        label_slice = labels[:, :, :, slice_id]
        # label_slice has shape [8, w, h, #slices], hence we separate es and ed and convert slice to multi-label
        # label_slice_es = convert_to_multiclass(label_slice[0:4])
        # label_slice_ed = convert_to_multiclass(label_slice[4:])
        pat_slice_rois[slice_id] = {}
        for cls_idx in np.arange(num_of_classes):
            if cls_idx not in bg_classes:
                label_slice_cls = label_slice[cls_idx]
                # returns BoundingBox object for the target roi
                gt_label_roi_cls = find_bbox_object(label_slice_cls, padding=0)
                pat_slice_rois[slice_id][cls_idx] = tuple((gt_label_roi_cls.slice_x, gt_label_roi_cls.slice_y))
    return pat_slice_rois


def get_label_areas(pat_slice_rois):
    """
    For a specific patient, get the labels_target_roi we calculated in the method above. The dictionary (per patient)
    contains a dictionary, with keys 0=ES and 1=ED. Each entry of this dict contains a tuple specifying the
    array slices of the target tissue regions in x and y direction.
    We use these to construct a BoundingBox object and compute the area-size of the target structure

    :param pat_slice_rois:
    :return: numpy array [2, #slice] with tissue label areas per slice (NOT PER CLASS)
    """
    # Note labels_target_roi is a dictionary with key=slice_id and hence length = #slices

    num_of_slices = len(pat_slice_rois)
    # 8 indices: as always for all eight tissue classes 0-3=ES and 4-7=ED
    pat_slice_areas = np.zeros((8, num_of_slices))
    for slice_id, slice_rois in pat_slice_rois.items():
        # slice_rois is again a dictionary with key=tissue class index AND value=tuple(slice_x, slice_y)
        for cls_idx in slice_rois.keys():
            roi_box_cls = BoundingBox(slice_rois[cls_idx][0], slice_rois[cls_idx][1])
            area_cls = roi_box_cls.width * roi_box_cls.height
            pat_slice_areas[cls_idx, slice_id] = area_cls
    return pat_slice_areas
