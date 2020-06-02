import numpy as np


def compute_tp_tn_fn_fp(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    tp_idx = result & reference
    fn = np.count_nonzero(~result & reference)
    fn_idx = ~result & reference
    tn = np.count_nonzero(~result & ~reference)
    tn_idx = ~result & ~reference
    fp = np.count_nonzero(result & ~reference)
    fp_idx = result & ~reference

    return tuple((tp, tp_idx)), tuple((fn, fn_idx)), tuple((tn, tn_idx)), tuple((fp, fp_idx))


def compute_detection_performance(result_dict, nbr_of_thresholds=40, range_threshold=[0, 1.]):

    """

    :param result_dict: dict [patient_id] of dict [frame_id] of dict ['pred_probs', 'gt_labels', 'gt_voxel_count']
                        of dict [#slices] <--- which are actually stored as strings e.g. '1' because we can't save
                        these dicts with numpy.savez
    :param nbr_of_thresholds:
    :param range_threshold:
    :return:
    """

    threshold_list = (np.linspace(range_threshold[0], range_threshold[1], nbr_of_thresholds)).tolist()
    mean_sensitivity, mean_precision, mean_fp_rate, mean_fp = [], [], [], []
    mean_reg_sensitivity, mean_reg_precision, mean_reg_fp_rate, mean_reg_fp = [], [], [], []
    perc_slices_with_errors, mean_detection_rate = [], []
    for i, threshold in enumerate(threshold_list):
        perc_slices_with_errors = []
        mean_num_regions = []
        # we collect the performance per patient for a specific threshold and then average for that threshold
        threshold_sensitivity, threshold_precision, threshold_fp_rate, threshold_fp = [], [], [], []
        region_sensitivity, region_precision, region_fp_rate, region_fp = [], [], [], []
        detection_rate = []
        for patient_id, cardiac_phases in result_dict.items():
            for frame_id in cardiac_phases.keys():

                pred_probs_slices = cardiac_phases[frame_id]['pred_probs']
                gt_labels_slices = result_dict[patient_id][frame_id]['gt_labels']
                gt_voxel_count_slices = result_dict[patient_id][frame_id]['gt_voxel_count']
                # store all labels for this patient
                pat_pred_labels = []
                pat_gt_labels = []
                pat_reg_pred_labels = []
                pat_reg_gt_labels = []
                pat_gt_voxel_count = 0  # ref of how many voxels we should detect in this volume
                pat_dt_voxel_count = 0  # how many do we detect in a volume
                pat_region_count = 0
                # Loop over slices
                for slice_id, slice_pred_probs in pred_probs_slices.items():
                    slice_gt_labels = gt_labels_slices[slice_id]
                    slice_gt_voxel_count = gt_voxel_count_slices[slice_id]
                    slice_pred_labels = np.zeros_like(slice_pred_probs)
                    # determine the labels for all regions/tiles in the slice
                    slice_pred_labels[slice_pred_probs >= threshold] = 1
                    # because we're evaluating slices, we determine ONE label based on all regions for that slice
                    pat_pred_labels.append(1 if np.count_nonzero(slice_pred_labels) != 0 else 0)
                    pat_gt_labels.append(1 if np.count_nonzero(slice_gt_labels) != 0 else 0)
                    pat_gt_voxel_count += np.sum(slice_gt_voxel_count)
                    pat_dt_voxel_count += np.sum(slice_gt_voxel_count * slice_pred_labels)
                    # if pat_pred_labels[-1] == 1:
                    pat_reg_pred_labels.extend(slice_pred_labels)
                    pat_reg_gt_labels.extend(slice_gt_labels)
                    pat_region_count += slice_gt_labels.shape[0]
                # End loop over slices
                mean_num_regions.append(pat_region_count)
                if pat_gt_voxel_count != 0:
                    detection_rate.append(pat_dt_voxel_count / pat_gt_voxel_count)
                else:
                    # there are no voxels to be detected. In this case we can ignore the detection rate although we may
                    # be have detected FP regions.
                    pass

                pat_gt_labels = np.array(pat_gt_labels)
                perc_slices_with_errors.append(np.count_nonzero(pat_gt_labels)/pat_gt_labels.shape[0])
                pat_pred_labels = np.array(pat_pred_labels)
                pat_gt_labels = np.array(pat_gt_labels)
                s_tp_2, s_fn_2, s_tn_2, s_fp_2 = compute_tp_tn_fn_fp(pat_pred_labels, pat_gt_labels)

                if (float(s_tp_2[0]) + s_fn_2[0]) != 0:
                    threshold_sensitivity.append((float(s_tp_2[0]) / (float(s_tp_2[0]) + s_fn_2[0])))
                else:
                    threshold_sensitivity.append(1)
                if (float(s_tp_2[0]) + s_fp_2[0]) != 0:
                    threshold_precision.append((float(s_tp_2[0]) / (float(s_tp_2[0]) + s_fp_2[0])))
                else:
                    threshold_precision.append(1.)
                if (s_tn_2[0] + s_fp_2[0]) != 0:
                    threshold_fp_rate.append((float(s_fp_2[0]) / (float(s_tn_2[0]) + s_fp_2[0])))
                else:
                    threshold_fp_rate.append(1)
                threshold_fp.append(s_fp_2[0])

                # same for regions
                pat_reg_pred_labels = np.array(pat_reg_pred_labels)
                pat_reg_gt_labels = np.array(pat_reg_gt_labels)
                # print(type(pat_reg_pred_labels), pat_reg_pred_labels.shape, type(pat_reg_gt_labels),
                # pat_reg_gt_labels.shape)
                r_tp_2, r_fn_2, r_tn_2, r_fp_2 = compute_tp_tn_fn_fp(pat_reg_pred_labels, pat_reg_gt_labels)

                if (r_tp_2[0] + r_fn_2[0]) != 0:
                    region_sensitivity.append((float(r_tp_2[0]) / (float(r_tp_2[0]) + r_fn_2[0])))
                else:
                    region_sensitivity.append(1)
                if (r_tp_2[0] + r_fp_2[0]) != 0:
                    region_precision.append((float(r_tp_2[0]) / (float(r_tp_2[0]) + r_fp_2[0])))
                else:
                    region_precision.append(1)
                if (r_tn_2[0] + r_fp_2[0]) != 0:
                    region_fp_rate.append((float(r_fp_2[0]) / (float(r_tn_2[0]) + r_fp_2[0])))
                else:
                    region_fp_rate.append(1)
                region_fp.append(r_fp_2[0])
            # End loop over cardiac phases (ED, ES)
        # End loop patients
        mean_num_regions = np.mean(np.array(mean_num_regions))
        if len(threshold_sensitivity) != 0:
            mean_sensitivity.append(np.mean(np.array(threshold_sensitivity)))
        if len(threshold_precision) != 0:
            mean_precision.append(np.mean(np.array(threshold_precision)))
        if len(threshold_fp_rate) != 0:
            mean_fp_rate.append(np.mean(np.array(threshold_fp_rate)))
        if len(threshold_fp) != 0:
            mean_fp.append(np.mean(np.array(threshold_fp)))
        # regions
        if len(region_sensitivity) != 0:
            mean_reg_sensitivity.append(np.mean(np.array(region_sensitivity)))
        if len(region_precision) != 0:
            mean_reg_precision.append(np.mean(np.array(region_precision)))
        if len(region_fp_rate) != 0:
            mean_reg_fp_rate.append(np.mean(np.array(region_fp_rate)))
        if len(region_fp) != 0:
            mean_reg_fp.append(np.mean(np.array(region_fp)))
        # detection rate
        mean_detection_rate.append(np.mean(np.array(detection_rate)))
    print("Average % of slices with errors {:.2f}".format(np.mean(np.array(perc_slices_with_errors) * 100)))
    print("Average #regions/volume {:.2f}".format(mean_num_regions))

    return mean_sensitivity[::-1], mean_precision[::-1], mean_fp_rate[::-1], mean_fp[::-1], mean_reg_sensitivity[::-1], \
           mean_reg_precision[::-1], mean_reg_fp_rate[::-1], mean_reg_fp[::-1], mean_detection_rate[::-1], threshold_list


class RegionDetectionEvaluation(object):

    def __init__(self, pred_probs, gt_labels, slice_probs_dict, slice_labels_dict, exper_handler,
                 nbr_of_thresholds=50, base_apex_labels=None, loc=0, range_threshold=None):

        self.pred_probs = pred_probs
        self.gt_labels = gt_labels
        self.slice_probs_dict = slice_probs_dict
        self.slice_labels_dict = slice_labels_dict
        self.exper_handler = exper_handler
        self.nbr_of_thresholds = nbr_of_thresholds
        self.base_apex_labels = base_apex_labels
        # loc can have 2 values: 0=non-base-apex    1=base-apex
        self.loc = loc
        self.range_threshold = range_threshold

    def generate_auc_curves(self):
        pass

