import scipy.ndimage as scnd
import numpy as np

from scipy.ndimage.measurements import label as scipy_label
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def getLargestCC(segmentation):
    labels, count = scipy_label(segmentation, structure=np.ones((3, 3, 3)))
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+ 1
    return largestCC


def filter_connected_components(pred_labels, cls=None, verbose=False, rank=None):
    """

    :param pred_labels:
    :param cls: currently not in use, only for debug purposes
    :param verbose:
    :return:
    """
    if rank is None:
        rank = pred_labels.ndim

    if rank == 2:
        structure = [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
    elif rank == 3:
        structure = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    else:
        raise ValueError("Input has more than 3 dimensions which is not supported by this function")

    mask = pred_labels == 1

    cc_labels, n_comps = scnd.measurements.label(mask, structure=structure)
    if verbose:
        print(('INFO - Class {}: FCC {} components'.format(cls, n_comps)))
    if n_comps > 1:
        sel_comp = 0
        in_comp = 0
        # Find the largest connected component
        for i_comp in range(1, n_comps + 1):
            if np.sum(cc_labels == i_comp) > in_comp:
                sel_comp = i_comp
                in_comp = np.sum(cc_labels == i_comp)
        if verbose:
            print(('INFO - Class {}: Select component {}'.format(cls, sel_comp)))
        # set all pixels/voxels to 0 (background) if they don't belong to the LARGEST connected-component
        for i_comp in range(1, n_comps + 1):
            if i_comp != sel_comp:
                pred_labels[cc_labels == i_comp] = 0

    return pred_labels


def detect_largest_umap_areas_slice(slice_u_map, structure):

    binary_map = np.zeros(slice_u_map.shape).astype(np.bool)
    mask = slice_u_map != 0
    binary_map[mask] = True
    binary_structure = generate_binary_structure(binary_map.ndim, connectivity=2)
    bin_labels, num_of_objects = scipy_label(binary_map, binary_structure)
    blob_area_sizes = []
    blob_after_erosion_sizes = []
    if num_of_objects >= 1:
        for i in np.arange(1, num_of_objects + 1):
            binary_map = np.zeros(bin_labels.shape).astype(np.bool)
            binary_map[bin_labels == i] = 1
            blob_area_sizes.append(np.count_nonzero(binary_map))
            remaining_blob = binary_erosion(binary_map, structure)
            blob_after_erosion_sizes.append(np.count_nonzero(remaining_blob))
        blob_area_sizes = np.array(blob_area_sizes)
        blob_area_sizes[::-1].sort()
        # print(blob_after_erosion_sizes)
        blob_after_erosion_sizes = np.array(blob_after_erosion_sizes)
        blob_after_erosion_sizes[::-1].sort()

    return blob_area_sizes, blob_after_erosion_sizes


def detect_largest_umap_areas(filtered_umap, rank_structure=5, max_objects=5):
    """
    first detect connected structures and then use erosion to detect how "connected" the areas are.


    :param filtered_umap: uncertainty map for an mri slice of shape [2, width, height, #slices]
    :param rank_structure:
    :param max_objects:
    :return:
    """
    num_of_slices = filtered_umap.shape[3]
    size_detected_areas = np.zeros((2, num_of_slices, max_objects)).astype(np.int)
    size_areas_after_erosion = np.zeros((2, num_of_slices, max_objects)).astype(np.int)
    structure = np.ones((rank_structure, rank_structure))

    for slice_id in np.arange(num_of_slices):
        for phase in np.arange(2):
            slice_u_map = filtered_umap[phase, :, :, slice_id]
            blob_area_sizes, blob_after_erosion_sizes = detect_largest_umap_areas_slice(slice_u_map, structure)
            if len(blob_area_sizes) != 0:
                # print(blob_after_erosion_sizes)
                size_detected_areas[phase, slice_id, :blob_area_sizes.shape[0]] = blob_area_sizes[:max_objects]
                size_areas_after_erosion[phase, slice_id, :blob_after_erosion_sizes.shape[0]] = blob_after_erosion_sizes[:max_objects]
                # we only want the max_objects, hence truncate if necessary

    return size_areas_after_erosion, size_detected_areas
