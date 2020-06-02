import numpy as np

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
from scipy.ndimage import label
import math


def compute_slice_distance_transform_for_structure(reference, voxelspacing=None, connectivity=1, ):
    """
    We compute the
    :param reference: we assume shape [width, height] and binary encoding (1=voxels of target structure)
    :param voxelspacing:
    :param connectivity:
    :return:
    """
    reference = np.atleast_1d(reference.astype(np.bool))
    inside_obj_mask = np.zeros_like(reference).astype(np.bool)
    # binary structure
    footprint = generate_binary_structure(reference.ndim, connectivity)
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The reference array does not contain any binary object.')

    # result_border = np.logical_xor(result, binary_erosion(result, structure=footprint, iterations=1))
    # reference_border = np.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=1))

    inside_voxels_indices = binary_erosion(reference, structure=footprint, iterations=1)
    inside_obj_mask[inside_voxels_indices] = 1
    reference_border = np.logical_xor(reference, inside_voxels_indices)
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)

    return dt, inside_obj_mask, reference_border


def generate_adjusted_dt_map(labels, error_margins, voxelspacing=1.4):
    """
    Generate the distance transform maps for this volume

    :param labels: shape [#slices, nclasses, w, h]
    :param voxelspacing:
    :param error_margins: tuple (outside margin, inside margin)
    :return: distance transform maps for a particular tissue class per slice
    """
    error_margin_outside, error_margin_inside = error_margins

    dt_map = np.zeros_like(labels)
    num_of_slices, nclasses,  w, h = labels.shape
    for slice_id in np.arange(num_of_slices):
        # determine apex-base slice. we currently use a simple heuristic. first and last slice are base-apex
        for cls_idx in np.arange(1, nclasses):
            label_slice = labels[slice_id, cls_idx]
            if 0 != np.count_nonzero(label_slice):
                # print("processing slice {} cls {}".format(slice_id, cls_idx))
                dt, inside_obj_mask, surface_border = \
                    compute_slice_distance_transform_for_structure(label_slice, voxelspacing=voxelspacing)

                outside_obj_mask = np.logical_and(~inside_obj_mask, ~surface_border)
                # surface border distance is always ZERO
                dt[surface_border] = 0
                # inside structure: we subtract a fixed margin
                dt[inside_obj_mask] = dt[inside_obj_mask] - error_margin_inside
                # outside of target: structure we subtract a fixed margin.
                dt[outside_obj_mask] = dt[outside_obj_mask] - error_margin_outside
                dt[dt < 0] = 0
                dt_map[slice_id, cls_idx] = dt
            else:
                # print("WARNING!!!! slice {} cls {}".format(slice_id, cls_idx))
                penalty = math.sqrt((h**2 + w**2)) * voxelspacing
                dt_map[slice_id, cls_idx] = penalty
    return dt_map


def determine_target_voxels(auto_seg, reference, dt_map, cls_indices=None, bg_classes=[0, 4]):
    """
        Parameters:
        auto_seg [num_of_classes, w, h, #slices]
        reference [num_of_classes, w, h, #slices]
        dt_map (distance transfer) [num_of_classes, w, h, #slices]
        cls_indices (tissue class indices that we process) e.g. for ACDC [1, 2, 3, 5, 6, 7]
        In general we assume that 0=background class
    """
    num_of_classes, w, h, num_of_slices = auto_seg.shape
    target_areas = np.zeros_like(auto_seg)
    if cls_indices is None:
        cls_indices = list(np.arange(1, num_of_classes))
        # remove bg-class for ES
        for rm_idx in bg_classes:
            if rm_idx in cls_indices:
                cls_indices.remove(rm_idx)
    else:
        if not isinstance(cls_indices, list):
            cls_indices = [cls_indices]
    for cls_idx in cls_indices:
        for slice_id in np.arange(num_of_slices):
            auto_seg_slice = auto_seg[cls_idx, :, :, slice_id]
            reference_slice = reference[cls_idx, :, :, slice_id]
            dt_map_slice = (dt_map[cls_idx, :, :, slice_id] > 0).astype(np.bool)
            # for the proposal roi's we use the uncertainty info and the automatic segmentation mask
            # for the target roi's we use the reference & automatic labels (for training only)
            seg_errors = auto_seg_slice != reference_slice
            roi_target_map = np.logical_and(dt_map_slice, seg_errors)
            # filter single voxels
            # 7-1-2019 disabled because seems to make results slightly worse
            # roi_target_map = filter_single_voxels(roi_target_map)
            target_areas[cls_idx, :, :, slice_id] = roi_target_map

    return target_areas


class FilterSegErrors(object):

    min_size_myo = 10  # 20
    min_size_rv = 10  # 40
    min_size_lv = 10  # 40
    min_tissue_size = [None, min_size_rv, min_size_myo, min_size_lv]

    def __init__(self, seg_errors, dt_map, apex_base_slices):
        """
        We assume this object is only used in case the object contain ONE cardiac phase

        Parameters:
        pred_labels [#slices, num_of_classes, w, h]
        seg_errors [#slices, num_of_classes, w, h]  NOTE: BG classes=index 0 has NO seg-errors
        dt_map (distance transfer) [#slices, num_of_classes, w, h,]
        apex_base_slices: dict with 'A': slice id apex, 'B': slice id base

        """
        self.target_voxels = np.zeros_like(seg_errors)
        self.seg_errors = seg_errors
        self.dt_map = dt_map
        self.apex_slice_id, self.base_slice_id = apex_base_slices['A'], apex_base_slices['B']
        self.num_classes, self.num_of_slices = seg_errors.shape[1], seg_errors.shape[0]
        self.cls_indices = np.arange(1, self.num_classes)
        self.do_filter()

    def get_filtered_errors(self, do_reduce=False):
        # shape of self.target_voxels is [#slices, nclasses, y, x]
        if do_reduce:
            # merge classes, returns [z, y, x]
            return np.any(self.target_voxels, axis=1)
        else:
            return self.target_voxels

    def do_filter(self):
        for cls_idx in self.cls_indices:
            for slice_id in np.arange(self.num_of_slices):
                slice_error = self.seg_errors[slice_id, cls_idx]
                dt_map_slice = (self.dt_map[slice_id, cls_idx] > 0).astype(np.bool)
                if slice_id == self.apex_slice_id:
                    # in case we're dealing with an apex slice, we keep all errors if they consist of at least
                    self.target_voxels[slice_id, cls_idx] = self.filter_on_size(slice_error, min_size=15)
                else:
                    roi_target_map = np.logical_and(dt_map_slice, slice_error)
                    if np.count_nonzero(roi_target_map) != 0:
                        self.target_voxels[slice_id, cls_idx] = self.filter_on_size(slice_error,
                                                                                    filtered_seg_errors=roi_target_map,
                                                                                    min_size=self.min_tissue_size[
                                                                                        cls_idx], cls_idx=cls_idx)

    @staticmethod
    def filter_on_size(binary_labels, min_size=10, filtered_seg_errors=None, cls_idx=None):
        # bstructure = generate_binary_structure(2, 2)
        new_label_slice = np.zeros_like(binary_labels)
        cc_labels, n_comps = label(binary_labels)
        for i_comp in np.arange(1, n_comps + 1):
            comp_mask = cc_labels == i_comp
            if filtered_seg_errors is not None:
                # if the filtered target errors are not part of the larger component then skip
                if np.count_nonzero(np.logical_and(filtered_seg_errors, comp_mask)) == 0:
                    # print("{} not part of filtered voxels {}".format(cls_idx, i_comp))
                    continue
            # print("processing comp {}".format(i_comp))
            comp_mask_size = np.count_nonzero(comp_mask)
            if comp_mask_size >= min_size:
                new_label_slice[cc_labels == i_comp] = 1
                # print("{} (>{}) part of filtered voxels {}".format(cls_idx, min_size, comp_mask_size))
        return new_label_slice
