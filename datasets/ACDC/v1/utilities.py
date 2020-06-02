import glob
import numpy as np
import os
import scipy.ndimage

from config.dcnn.general_setup import base_config as dcnn_config


def remove_padding(image):
    """
    Remove padding of images that we use for training
    We always assume that x, y dimension are last

    :param image:
    :return:
    """
    return image[...,dcnn_config.pad_size:-dcnn_config.pad_size,
                     dcnn_config.pad_size:-dcnn_config.pad_size]


def binarize_apex_base_indicator(slice_id, vector_length):
    b_vector = np.zeros(vector_length)
    b_vector[slice_id] = 1
    return b_vector


def binarize_acdc_labels(labels, classes=[0, 1, 2, 3]):
    """

    :param labels: have shape [w, h] OR [w, h, d] or [2, w, h, d]  type np.int
    :return: binarized labels [4, w, h] OR [4, w, h, d] OR [8, w, h, d] WHERE 0:4=ES and 4:8=ED
    """
    if labels.shape[0] == 2 and labels.ndim == 4:
        # shape is [2, w, h, d]
        # we are dealing with a combined ES/ED label array
        array_es = np.stack([(labels[0] == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        array_ed = np.stack([(labels[1] == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        binary_array = np.concatenate([array_es, array_ed], axis=0)
    else:
        if labels.ndim == 4:
            raise ValueError("ERROR - binarize acdc labels - shape of array is 4 but first dim != 2 (ES/ED)")
        elif labels.ndim == 2 or labels.ndim == 3:
            # shape is [w, h] OR [w, h, d]
            binary_array = np.stack([(labels == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        else:
            raise ValueError("ERROR - binarize labels acdc - Rank {} of array not supported".format(labels.ndim))

    return binary_array


def split_acdc_dataset(acdc_fold, num_of_pats=100, max_per_set=None) -> tuple((list, list)):
    allpatnumbers = np.arange(1, num_of_pats + 1)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)
    # We use fold4 - fold7 for a model trained with all patient studies:
    # There is no fold4-7, but we use it to select ALL patients. Setting when segmentatin 4d ACDC dataset
    # Test set will be limited to 25 patients
    if acdc_fold >= 4:
        training_nums = allpatnumbers
        validation_nums = allpatnumbers[foldmask == acdc_fold - 4]
    else:
        training_nums, validation_nums = allpatnumbers[foldmask != acdc_fold], allpatnumbers[foldmask == acdc_fold]
    if max_per_set is not None:
        training_nums = training_nums[:max_per_set]
        validation_nums = validation_nums[:max_per_set]

    return training_nums, validation_nums


def split_arvc_dataset(arvc_fold, length_dataset, max_per_set=25) -> list:
    """
    For now we just take a very dump rule to split the dataset. In the future we need to adjust this
    but we're currently only evaluating our segmentation model on the ARVC dataset, so we only split
    for performance reasons (otherwise dataset objects blows up)

    :param arvc_fold:
    :param max_per_set:
    :return:
    """
    if arvc_fold >= int(np.ceil(length_dataset / max_per_set)):
        raise ValueError("ERROR - split_arvc_fold - fold {} is to big for dataset with length"
                         " {} and {} samples per fold. Max is {}".format(arvc_fold,
                                                                         length_dataset,
                                                                         max_per_set,
                                                                         int(length_dataset / max_per_set) + 1))

    range_idx = np.arange(length_dataset)
    rest_length = length_dataset % max_per_set
    start_idx = arvc_fold * max_per_set
    stop_idx = (arvc_fold + 1) * max_per_set if arvc_fold < int(length_dataset / max_per_set) else \
        (arvc_fold * max_per_set) + rest_length
    print("fold_id {} (len={}): Start/Stop {}/{}".format(arvc_fold, length_dataset, start_idx, stop_idx))
    return range_idx[start_idx:stop_idx]


def crawl_dir(in_dir, load_func="load_itk", pattern="*.mhd", logger=None):
    """
    Searches for files that match the pattern-parameter and assumes that there also
    exist a <filename>.raw for this mhd file
    :param in_dir:
    :param load_func:
    :param pattern:
    :param logger:
    :return: python list with
    """
    im_arrays = []
    gt_im_arrays = []

    pattern = os.path.join(in_dir, pattern)
    for fname in glob.glob(pattern):
        mri_scan, origin, spacing = load_func(fname)
        logger.info("Loading {}".format(fname, ))
        im_arrays.append((mri_scan, origin, spacing))

    return im_arrays, gt_im_arrays


def rescale_image(img, perc_low=5, perc_high=95, axis=None):
    # flatten 3D image to 1D and determine percentiles for rescaling
    lower, upper = np.percentile(img, [perc_low, perc_high], axis=axis)
    # set new normalized image
    img = ((img.astype(float) - lower) / (upper - lower)).clip(0, 1)

    return img


def normalize_image(img, axis=None):
    img = (img - np.mean(img, axis=axis)) / np.std(img, axis=axis)
    return img


def apply_2d_zoom_3d(arr3d, spacing, new_spacing, order=1, do_blur=True, as_type=np.float32):
    """

    :param arr3d: [#slices, IH, IW]
    :param spacing: spacing has shape [#slices, IH, IW]
    :param new_vox_size: tuple(x, y)
    :param order: of B-spline
    :param do_blur: boolean (see below)
    :param as_type: return type of np array. We use this to indicate binary/integer labels which we'll round off
                    to prevent artifacts
    :return:
    """
    if len(spacing) > 2:
        spacing = spacing[int(len(spacing) - 2):]

    if len(new_spacing) > 2:
        new_spacing = new_spacing[int(len(new_spacing) - 2):]

    zoom = np.array(spacing, float) / new_spacing
    if do_blur:
        for z in range(arr3d.shape[0]):
            sigma = .25 / zoom
            arr3d[z, :, :] = scipy.ndimage.gaussian_filter(arr3d[z, :, :], sigma)

    resized_img = scipy.ndimage.interpolation.zoom(arr3d, tuple((1,)) + tuple(zoom), order=order)
    if as_type == np.int:
        # binary/integer labels
        resized_img = np.round(resized_img).astype(as_type)
    return resized_img


def apply_2d_zoom_4d(arr4d, spacing, new_spacing=(1.4, 1.4)):
    """

    :param arr4d: numpy tensor [#phases, z, y, x]
    :param spacing: tuple [z, y, x] with float for each of the three dims (only 3, not for #phases)
    :param new_spacing: mm new vox size
    :return: resampled volume
    """
    if len(spacing) > 2:
        spacing = spacing[int(len(spacing) - 2):]
    if len(new_spacing) > 2:
        new_spacing = new_spacing[int(len(new_spacing) - 2):]

    zoom = np.array(spacing, float) / new_spacing
    # loop over cardiac phases
    for idx in range(arr4d.shape[0]):
        # loop over volume depth (z)
        for jdx in range(arr4d.shape[1]):
            sigma = .25 / zoom
            arr4d[idx, jdx] = scipy.ndimage.gaussian_filter(arr4d[idx, jdx], sigma)
    return scipy.ndimage.interpolation.zoom(arr4d, (1, 1) + tuple(zoom), order=1)


def apply_2d_zoom_pred_labels(arr4d, spacing, new_spacing=(1.4, 1.4)):
    """
        Resampling 4d numpy arrays of type np.int

        :param arr4d: numpy tensor [#phases, z, y, x]
        :param spacing: numpy array [z, y, x] we can handle length 3 or 4 (so with frame number)
        :param new_spacing: numpy array [z, y, x]
        :return: resampled volume
    """
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    if not arr4d.dtype == np.int:
        arr4d = arr4d.astype(np.int)
    if len(spacing) > 2:
        spacing = spacing[int(len(spacing) - 2):]

    if len(new_spacing) > 2:
        new_spacing = new_spacing[int(len(new_spacing) - 2):]

    zoom = spacing / new_spacing
    zoom = (1, 1) + tuple(zoom)
    #  order of polynomial used to resample. For integer labels we use 0, nearest neighbor
    resized_img = scipy.ndimage.interpolation.zoom(arr4d, zoom, order=0)
    return np.round(resized_img).astype(np.int)
