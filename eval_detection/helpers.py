from scipy.stats import mannwhitneyu
import numpy as np


def compute_p_values(result_obj1, result_obj2, perf_type):
    l_vectors_base, _, _ = split_by_tissue_class(result_obj1, perf_type)
    l_vectors_correct_all, _, _ = split_by_tissue_class(result_obj2, perf_type)
    # l_vectors_dict is return object from split_by_tissue_class
    p_values = {}
    for cardiac_phase, perf_obj1 in l_vectors_base.items():
        perf_obj2 = l_vectors_correct_all[cardiac_phase]
        nclasses = len(perf_obj1)
        p_values[cardiac_phase] = np.zeros(nclasses)
        for cls_idx in np.arange(1, nclasses):
            _, p_values[cardiac_phase][cls_idx] = mannwhitneyu(perf_obj1[cls_idx], perf_obj2[cls_idx])

    return p_values


def split_by_tissue_class(result_object, perf_type):
    """

    :param result_object: is of object TestResult()
    :param perf_type:
    :return:
    """
    if perf_type == "HD":
        archive_es, archive_ed = 'hd_es', 'hd_ed'
    elif perf_type == "DSC":
        archive_es, archive_ed = 'dice_es', 'dice_ed'
    vol_stats_es, vol_stats_ed = result_object[archive_es], result_object[archive_ed]

    # s_stats_es and ed have shape [#patients, nclasses]
    # we're going to split them into separate arrays per tissue class
    l_vectors = {}
    l_vectors['ES'] = [np.squeeze(arr) for arr in np.hsplit(vol_stats_es, vol_stats_es.shape[1])]
    l_vectors['ED'] = [np.squeeze(arr) for arr in np.hsplit(vol_stats_ed, vol_stats_ed.shape[1])]
    return l_vectors, vol_stats_es, vol_stats_ed