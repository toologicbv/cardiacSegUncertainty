import os
import numpy as np
from scipy.stats import binned_statistic
from datasets.ACDC.get_data import load_data
from tqdm import tqdm
import math


def compute_expected_calibration_error(pred_probs_cls, acc_per_bin, counts_pred_probs, num_of_bins=10):
    """

    :param pred_probs_cls: numpy flat array containing all softmax probabilities for this class/patient
    :param acc_per_bin: numpy array size [num_of_bins] specifying the accuracy for each probability bin
    :param counts_pred_probs: np array size [num_of_bin] specifying the counts in each prob bin
    :param num_of_bins:

    :return: ECE (expected calibration error), scalar
    """
    _ = np.seterr(divide='ignore')
    mean_prob_in_bin, _, _ = binned_statistic(pred_probs_cls, pred_probs_cls, statistic='mean',
                                              bins=num_of_bins, range=(0, 1.))
    mean_prob_in_bin = np.nan_to_num(mean_prob_in_bin)
    # weighted average: numerator=number of probs in each of the bins,
    bin_scaler = counts_pred_probs * 1./np.sum(counts_pred_probs)
    # print("acc & conf")
    # print(acc_per_bin, mean_prob_in_bin)

    ece = np.sum(bin_scaler * np.abs(acc_per_bin - mean_prob_in_bin))
    # print("ECE ", ece)
    # print(mean_prob_in_bin)
    return ece


def compute_calibration(src_data_path, cardiac_phase='all', patient_id=None, mc_dropout=False, num_of_bins=10,
                              do_save=False, with_bg=False, nclasses=4, output_dir=None):
    """
    Compute the ingredients we need for the ECE (Expected Calibration Error) and the Reliability Diagrams
    Taken from paper https://arxiv.org/abs/1706.04599

    :param src_data_path:
    :param patient_id:
    :param mc_dropout:
    :param num_of_bins:
    :param with_bg: include background class. By default we don't include the class (because we also don't report
                    results on segmentation performance for this class).
    :param do_save:
    :return:
    """
    if with_bg:
        class_range = np.arange(nclasses)
        print("INFO - WITH background class")
    else:
        class_range = np.arange(1, nclasses)

    if cardiac_phase.lower() == 'all':
        cardiac_phases = ['ED', 'ES']
    else:
        cardiac_phases = [cardiac_phase]
    # prepare: get probability maps and test_set which contains
    # prob_bins: contains bin-edges for the probabilities (default 10)
    prob_bins = np.linspace(0, 1, num_of_bins + 1)
    # acc_bins: accuracy per bin. numerator="# of positive predicted class labels for this bin";
    #                             denominator="# of predictions in this bin"
    acc_bins = np.zeros((nclasses, num_of_bins))
    probs_per_bin = np.zeros((nclasses, num_of_bins))
    probs_per_bin_denom = np.zeros((nclasses, num_of_bins))
    acc_bins_used = np.zeros((nclasses, num_of_bins))
    mean_counts_per_bin = np.zeros((nclasses, num_of_bins))
    # confidence measure = mean predicted probability in this bin
    ece_per_class = np.zeros(nclasses)
    ece_counts_per_class = np.zeros(nclasses)
    print("INFO - Loading probabilities and predictions from {}".format(src_data_path))

    for c_phase in cardiac_phases:
        data_dict = load_data(src_data_path, c_phase, ['pred_labels', 'pred_probs', 'ref_labels'], mc_dropout)
        if patient_id is not None:
            patients = [patient_id]
        else:
            patients = data_dict['ref_labels'].keys()
        for p_id in tqdm(patients, desc="Processing softmax predictions"):
            if p_id in ['patient029']:
                print("WARNING - skipping patient {}".format(p_id))
                continue
            gt_labels = data_dict['ref_labels'][p_id]
            # pred_labels = exper_handler.pred_labels[p_id]
            pred_probs = data_dict['pred_probs'][p_id]
            for cls_idx in class_range:
                gt_labels_cls = gt_labels[:, cls_idx].flatten()
                pred_probs_cls = pred_probs[:, cls_idx].flatten()
                gt_labels_cls = np.atleast_1d(gt_labels_cls.astype(np.bool))
                # determine indices of the positive/class voxels
                pos_voxels_idx = gt_labels_cls == 1
                # get all predicted probabilities that "predicted" correctly the pos-class label
                pred_probs_cls_cor = pred_probs_cls[pos_voxels_idx]
                counts_pred_probs, _ = np.histogram(pred_probs_cls, bins=prob_bins)
                cls_idx_probs_per_bin = np.digitize(pred_probs_cls, bins=prob_bins)
                # bin the predicted probabilities with correct predictions (class labels)
                counts_pred_probs_cor, _ = np.histogram(pred_probs_cls_cor, bins=prob_bins)
                acc_per_bin = np.zeros(num_of_bins)
                for bin_idx in np.arange(len(acc_per_bin)):
                    # do the stuff to compute the conf(B_m) metric from the Guo & Pleiss paper
                    cls_probs_per_bin = pred_probs_cls[cls_idx_probs_per_bin == bin_idx + 1]
                    probs_per_bin[cls_idx, bin_idx] += np.sum(cls_probs_per_bin)
                    # add total number of voxels in this bin (positive + others)
                    probs_per_bin_denom[cls_idx, bin_idx] += cls_probs_per_bin.shape[0]
                    # do the stuff to compute the acc(B_m) metric from the Guo&Pleiss paper
                    if counts_pred_probs[bin_idx] != 0:
                        acc_per_bin[bin_idx] = counts_pred_probs_cor[bin_idx] * 1./counts_pred_probs[bin_idx]
                        acc_bins[cls_idx, bin_idx] += acc_per_bin[bin_idx]

                        if counts_pred_probs_cor[bin_idx] != 0:
                            acc_bins_used[cls_idx, bin_idx] += 1
                    else:
                        acc_bins[cls_idx, bin_idx] += 0
                # compute ECE for this patient/class
                ece = compute_expected_calibration_error(pred_probs_cls, acc_per_bin, counts_pred_probs,
                                                         num_of_bins=num_of_bins)
                ece_per_class[cls_idx] += ece
                ece_counts_per_class[cls_idx] += 1
                mean_counts_per_bin += counts_pred_probs

    # print(acc_bins_used)
    print(ece_per_class)
    print(ece_counts_per_class)
    for cls_idx in np.arange(nclasses):
        for bin_idx in np.arange(len(prob_bins[1:])):
            if acc_bins_used[cls_idx, bin_idx] != 0:
                acc_bins[cls_idx, bin_idx] *= 1./acc_bins_used[cls_idx, bin_idx]
            if probs_per_bin_denom[cls_idx, bin_idx] != 0:
                probs_per_bin[cls_idx, bin_idx] *= 1./probs_per_bin_denom[cls_idx, bin_idx]

    # compute final mean ECE value per class, omit the BACKGROUND class
    mean_ece_per_class = np.nan_to_num(np.divide(ece_per_class, ece_counts_per_class))
    # compute mean counts per probability bin
    mean_counts_per_bin *= mean_counts_per_bin
    if do_save:
        try:
            e_suffix = ""
            if with_bg:
                e_suffix = "_wbg"

            file_name = "calibration_" + e_suffix + ".npz"
            file_name = os.path.join(output_dir, file_name)
            np.savez(file_name, prob_bins=prob_bins, acc_bins=acc_bins, mean_ece_per_class=mean_ece_per_class,
                     probs_per_bin=probs_per_bin)
            print(("INFO - Successfully saved numpy arrays to location {}".format(file_name)))
        except IOError:
            raise IOError("ERROR - can't save numpy arrays to location {}".format(file_name))

    return {'prob_bin_edges': prob_bins, 'acc_per_bin': acc_bins, 'mean_ece_per_class': mean_ece_per_class,
            'probs_per_bin': probs_per_bin, 'load_dir': src_data_path, 'nclasses': nclasses,
            'cardiac_phase': cardiac_phase, 'mc_dropout': mc_dropout, 'with_bg': with_bg}


if __name__ == "__main__":
    # dcnn_mc_daugbrierv2
    # drn_mc_brierv2_cyclic
    input_dir = "/home/jorg/expers/acdc/dcnn_mc_daugbrierv2"
    # data_dict = load_data(src_data_path=input_dir, cardiac_phase="ES", mc_dropout=False)
    # print(len(data_dict['ref_labels']), data_dict['ref_labels']['patient016'].shape, data_dict['pred_labels']['patient016'].shape)
    prob_bins, acc_bins, mean_ece_per_class, probs_per_bin = \
        compute_calibration(input_dir, "ES", patient_id=None, mc_dropout=False)
    print(acc_bins)
