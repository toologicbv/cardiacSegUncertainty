import os
import numpy as np
from tqdm import tqdm
from utils.common import detect_seg_errors
import copy
from datasets.ACDC.get_data import get_pred_labels, get_umaps, get_detector_target_labels, get_ref_labels
from utils.medpy_metrics import dc, hd
from utils.common import one_hot_encoding, loadExperimentSettings
from utils.box_utils import find_bbox_object
from datasets.data_config import get_config
import datasets.ACDC.data


def get_seg_errors_mask(pred_labels, ref_labels):
    # pred_labels: [#slices, nclasses, x, y]
    # ref_labels have shape: [#slices, x, y]
    err_indices = np.zeros_like(ref_labels)

    for cls_idx in np.arange(0, pred_labels.shape[1]):
        pr_labels = pred_labels[:, cls_idx]
        err_indices[(pr_labels == 1) != (ref_labels == cls_idx)] = 1

    return err_indices.astype(np.bool)


def generate_thresholds(pred_labels, ref_labels, umap):
    err_indices = get_seg_errors_mask(pred_labels, ref_labels)
    umap_err = umap[err_indices]
    percentiles = [np.percentile(umap_err, p) for p in np.arange(1, 101)]
    percentiles = [0] + percentiles
    return percentiles


class SelectiveClassification(object):

    def __init__(self, src_data_path, cardiac_phase, num_of_thresholds=30, verbose=False,
                 patients=None, mc_dropout=False, dataset="ACDC", dt_config_id=None):

        self.src_data_path = src_data_path
        self.pred_labels, self.umaps = None, None
        self.mc_dropout = mc_dropout
        self.type_of_map = 'bmap' if mc_dropout else "emap"
        # TODO IMPORTANT: if dt_config_id is not None then we actually use the detection labels (filtered segmentation
        # errors) as constrained for the mis-predicted seg errors that we "correct" based on the voxel uncertainties
        self.dt_config_id = dt_config_id
        # Important: how many measurements are we performing between rejection 0 voxels and all voxels
        self.num_of_thresholds = num_of_thresholds + 1
        self.x_coverages = None
        self.patient_coverages = None
        self.mean_errors = None
        self.seg_errors = None
        self.use_cropped = False  # whether or not to crop the image to the ground truth region (with padding=10)
        self.patient_dsc = None
        self.patient_hd = None
        self.verbose = verbose
        self.patients = patients
        self.cardiac_phase = cardiac_phase
        self.dta_settings = get_config(dataset)

        # for each patient data, store the optimal C-R curve that could have been achieved
        # based on Geifman paper "Boosting uncertainty estimation..."
        self.optimal_curve = []
        self.save_output_dir = os.path.expanduser(self.src_data_path)
        self._prepare()

    def _prepare(self):
        self.pred_labels = get_pred_labels(self.src_data_path, self.cardiac_phase, mc_dropout=self.mc_dropout,
                                           one_hot=True, meta_info=True)
        self.umaps = get_umaps(self.src_data_path, self.cardiac_phase, mc_dropout=self.mc_dropout)
        if self.dt_config_id is not None:
            self.dt_labels = get_detector_target_labels(self.src_data_path, self.cardiac_phase, self.dt_config_id)

    def generate(self, save_example_slices=False, use_cropped=False):
        self.use_cropped = use_cropped
        self.optimal_curve = []
        self.x_coverages = np.linspace(0, 1, self.num_of_thresholds)
        self.mean_errors = np.empty((0, self.num_of_thresholds))
        self.seg_errors = np.empty((0, self.num_of_thresholds))
        self.patient_coverages = np.empty((0, self.num_of_thresholds))
        self.patient_dsc = np.empty((0, self.num_of_thresholds, 4))
        self.patient_hd = np.empty((0, self.num_of_thresholds, 4))
        for patient_id in tqdm(self.pred_labels.keys(), desc="Processing patient volumes"):
                if patient_id == "patient029":
                    print("WARNING - Skipping {} because re-sampled size of pred_labels doesn't fit resampled umap".format(patient_id))
                    continue
                if self.patients is not None and patient_id not in self.patients:
                    continue
                labels_dict, _ = get_ref_labels(cardiac_phase=self.cardiac_phase, patient_id=patient_id,
                                                one_hot=False, limited_load=False, resample=False)
                labels = labels_dict[patient_id]
                patient_coverages = []
                patient_error_perc = []
                patient_dices = []
                patient_hds = []
                patient_seg_errors = []
                # pred_labels is one-hot-encoded and has shape [z, n_classes, y, x]
                pred_labels = self.pred_labels[patient_id]['pred_labels']
                spacing = self.pred_labels[patient_id]['spacing']
                umap = self.umaps[patient_id]['umap']
                pat_thresholds = generate_thresholds(pred_labels, labels, umap)
                # print(pat_thresholds)
                example_corrected_slice = np.zeros((self.num_of_thresholds, 2, pred_labels.shape[0],
                                                    pred_labels.shape[2], pred_labels.shape[3]))
                map_min, map_max = np.min(umap), np.max(umap)
                # print("Min/Max map ", patient_id, map_min, map_max)
                # pred_probs has shape [num_of_classes, w, h], but we need the maximum prob per class to compute
                # the coverage
                # pred_probs = np.max(self.data_handler.pred_probs[patient_id], axis=0)
                num_of_slices, w, h = labels.shape
                errors_slices = np.zeros(num_of_slices)
                sel_errors_slices = np.zeros(num_of_slices)
                roi_areas = np.zeros(num_of_slices)
                roi_slices = []
                cov_slices = np.zeros(num_of_slices)
                optimal_error = np.zeros(num_of_slices)
                start = True
                # for t, threshold in enumerate(np.linspace(0, map_max, self.num_of_thresholds)):
                for t, threshold in enumerate(pat_thresholds):
                    # print("INFO - Applying threshold {:.4f}".format(threshold))
                    new_pred_labels = np.zeros_like(pred_labels)
                    for slice_id in np.arange(num_of_slices):
                        pred_labels_slice = copy.deepcopy(pred_labels[slice_id, :, :, :])
                        gt_labels_slice = labels[slice_id, :, :]
                        dt_label_slice = None
                        if self.dt_config_id is not None:
                            dt_label_slice = self.dt_labels[patient_id][slice_id]
                        map_slice = umap[slice_id, :, :]

                        if start:
                            # we do this only once for each slice (i.e. for the first threshold).
                            # Remember: detect_seg_errors returns binary mask indicating errors for slice voxels
                            # hence returns [y, x] shape.
                            seg_errors_slice = detect_seg_errors(gt_labels_slice, pred_labels_slice,
                                                                 is_multi_class=True)
                            num_of_seg_errors = np.count_nonzero(seg_errors_slice)
                            errors_slices[slice_id] = num_of_seg_errors
                            if use_cropped:
                                roi_slice_bbox = find_bbox_object(gt_labels_slice, padding=10)
                                roi_areas[slice_id] = roi_slice_bbox.area
                                roi_slices.append(roi_slice_bbox)
                                # if the slice doesn't contain any gt labels the roi is empty and we just
                                # consider the whole slice instead of the cropped size
                                if roi_slice_bbox.area != 0:
                                    optimal_error[slice_id] = num_of_seg_errors * 1./roi_slice_bbox.area  # 1./(w * h)
                                else:
                                    # print("WARNING - no roi box area ", slice_id, roi_slice_bbox.slice_x, roi_slice_bbox.slice_y)
                                    optimal_error[slice_id] = num_of_seg_errors * 1./(w * h)
                                    roi_areas[slice_id] = w * h
                            else:
                                optimal_error[slice_id] = num_of_seg_errors * 1. / (w * h)
                                # set voxels equal/above threshold to gt label
                        uncertain_voxels_idx = map_slice >= threshold
                        # print("#Uncertain ", np.sum(map_slice < 0.), threshold, np.count_nonzero(uncertain_voxels_idx))
                        if self.dt_config_id is not None:
                            uncertain_voxels_idx = SelectiveClassification.filter_uncertain_voxels(uncertain_voxels_idx, dt_label_slice)
                        # Sum non-rejected probability mass
                        # cov_slices[slice_id] = np.sum(pred_probs_slice[~uncertain_voxels_idx])
                        # According to Greifman the empirical coverage is just the number of voxels not-rejected.
                        pred_labels_slice = self._set_selective_voxels(pred_labels_slice, gt_labels_slice,
                                                                       uncertain_voxels_idx)
                        if use_cropped:
                            # IMPORTANT: we have slices that do not contain any target segmentations so we have no bounding box
                            # in order to make sure computation of coverage is correct, we set cov_slices[slice_id] in those cases
                            # equal to the area of the complete slice (see else:). This is the same we do above when initially
                            # computing the area (in voxel count) of the slice region we're referring
                            if roi_slices[slice_id].area != 0:
                                unc_voxels = map_slice[roi_slices[slice_id].slice_x, roi_slices[slice_id].slice_y] >= threshold
                                cov_slices[slice_id] = np.sum(~unc_voxels)
                            else:
                                # print("Coverage - no roi box area ", slice_id, roi_slices[slice_id].slice_x, roi_slices[slice_id].slice_y)
                                cov_slices[slice_id] = roi_areas[slice_id]
                        else:
                            # using the complete slice:
                            cov_slices[slice_id] = np.sum(~uncertain_voxels_idx)

                        new_pred_labels[slice_id, :, :, :] = pred_labels_slice
                        # returns slice in which the incorrect labeled voxels are indicated by the gt class, using
                        # multi-class indices {1...nclass}
                        seg_errors_slice = detect_seg_errors(gt_labels_slice, pred_labels_slice,
                                                             is_multi_class=True)

                        sel_errors_slices[slice_id] = np.count_nonzero(seg_errors_slice)
                        if False and self.verbose:
                            print("INFO - Processing slice {}: {} : {}".format(slice_id + 1, errors_slices[slice_id],
                                                                               sel_errors_slices[slice_id]))
                        # sometimes we want to save an example of how errors in slice disappear with decreased
                        # uncertainty threshold
                        if save_example_slices and (slice_id == 0 or slice_id == 1):
                            ref_labels_slice = one_hot_encoding(labels[slice_id])
                            example_corrected_slice[t, slice_id] = new_pred_labels[slice_id]
                    # end patient/threshold
                    start = False
                    pat_dices, pat_hds = self.compute_metrics(new_pred_labels, labels, spacing)
                    # print("\t Dice@{:.3f} {:.3f}".format(threshold, np.mean(pat_dices)))
                    error_perc = np.sum(sel_errors_slices) / float(np.sum(errors_slices))

                    if use_cropped:
                        coverage = np.mean(cov_slices * 1./roi_areas)
                    else:
                        # using the total slice:
                        coverage = np.mean(cov_slices * 1. / (w * h))

                    patient_coverages.append(coverage)
                    patient_error_perc.append(error_perc)
                    patient_dices.append(pat_dices)
                    patient_hds.append(pat_hds)
                    # sum seg errors for this threshold over the compete patient volume
                    patient_seg_errors.append(np.sum(sel_errors_slices))
                    if self.verbose:
                        print("INFO - patient {}: tau {:.7f}: DSC {:.3f} HD {:.3f} "
                              "error% {:.3f} #seg-errors (after corr.) {} Coverage% {:.3f}".format(patient_id, threshold,
                                                                                        np.mean(pat_dices[1:]),
                                                                      np.mean(pat_hds[1:]),
                                                                      error_perc, patient_seg_errors[-1], coverage))

                # End: for a patient applying all thresholds
                self.optimal_curve.append(np.mean(optimal_error))
                # prepare for interplolation, reverse ordering
                patient_coverages = np.array(patient_coverages)
                patient_error_perc = np.array(patient_error_perc)
                patient_seg_errors = np.array(patient_seg_errors)
                # results in an array of shape [#thresholds, 4]
                patient_dices = np.stack(patient_dices)
                patient_hds = np.stack(patient_hds)
                # error perc
                first_value = patient_error_perc[0]
                patient_error_perc = np.interp(self.x_coverages[1:], patient_coverages[1:], patient_error_perc[1:])
                patient_error_perc = np.concatenate((np.array([first_value]), patient_error_perc))
                self.mean_errors = np.vstack((self.mean_errors, patient_error_perc)) if self.mean_errors.size else patient_error_perc
                # seg errors
                first_value = patient_seg_errors[0]
                patient_seg_errors = np.interp(self.x_coverages[1:], patient_coverages[1:], patient_seg_errors[1:])
                patient_seg_errors = np.concatenate((np.array([first_value]), patient_seg_errors))
                self.seg_errors = np.vstack(
                    (self.seg_errors, patient_seg_errors)) if self.seg_errors.size else patient_seg_errors
                # Interpolate dice scores. results in array shape [#patients, #thresholds, 4]
                threshold_pat_dices = copy.deepcopy(patient_dices)
                # DSC
                first_value = patient_dices[0]
                patient_dices = SelectiveClassification.interpolate_dice(self.x_coverages[1:], patient_coverages[1:], patient_dices[1:])
                patient_dices = np.concatenate((np.array([first_value]), patient_dices))
                patient_dices = np.expand_dims(patient_dices, axis=0)
                self.patient_dsc = np.vstack((self.patient_dsc, patient_dices)) if self.patient_dsc.size else patient_dices
                # HD
                first_value = patient_hds[0]
                patient_hds = SelectiveClassification.interpolate_dice(self.x_coverages[1:], patient_coverages[1:], patient_hds[1:])
                patient_hds = np.concatenate((np.array([first_value]), patient_hds))
                patient_hds = np.expand_dims(patient_hds, axis=0)
                self.patient_hd = np.vstack((self.patient_hd, patient_hds)) if self.patient_hd.size else patient_hds

                if save_example_slices:
                    used_thresholds = np.linspace(0, map_max, self.num_of_thresholds).astype(np.float)
                    self._save_corrected_labels(patient_id, example_corrected_slice, used_thresholds, patient_dices,
                                                threshold_pat_dices)
        self.optimal_curve = np.mean(self.optimal_curve)
        del patient_coverages
        del patient_error_perc

    @staticmethod
    def compute_metrics(pred_labels, ref_labels, spacing=None):
        # pred_labels has shape [z, n_classes, y, x] and ref_labels [z, y, x] or [z, n_classes, y, x]
        if ref_labels.ndim == 3:
            ref_labels = one_hot_encoding(ref_labels)

        num_classes = ref_labels.shape[1]
        dices = np.zeros(num_classes)
        if spacing is not None:
            hds = np.zeros(num_classes)

        for cls in np.arange(1, num_classes):
            dices[cls] = dc(pred_labels[:, cls], ref_labels[:, cls])
            if spacing is not None:
                if np.count_nonzero(pred_labels[:, cls]) != 0 and np.count_nonzero(ref_labels[:, cls]) != 0:
                    hds[cls] = hd(pred_labels[:, cls], ref_labels[:, cls])
        return dices, hds

    @staticmethod
    def interpolate_dice(coverages, patient_coverages, patient_dice):

        interp_dice = np.zeros((len(coverages), 4))
        for cls_idx in range(patient_dice.shape[1]):
            interp_dice[:, cls_idx] = np.interp(coverages, patient_coverages, patient_dice[:, cls_idx])
        return interp_dice

    def _save_corrected_labels(self, patient_id, corrected_slices, thresholds, cov_pat_dice, thre_pat_dice):
        outdir = os.path.join(os.path.join(self.src_data_path, 'figures'), patient_id)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        fname = os.path.join(outdir,
                             patient_id + "_mc" + self.mc_dropout + "_corr_labels.npz")
        np.savez(fname, new_pred_labels=corrected_slices, thresholds=thresholds, cov_dice=np.squeeze(cov_pat_dice),
                 threshold_dice=thre_pat_dice,
                 coverages=self.x_coverages, cardiac_phase=self.cardiac_phase, loss_function=self.loss_function)

    @staticmethod
    def filter_uncertain_voxels(high_uvalue_indices, dt_labels):
        num_classes = dt_labels.shape[0]
        filtered_uncertain_voxels = np.zeros_like(high_uvalue_indices).astype(np.bool)
        for cls in np.arange(num_classes):
            correct_voxels = np.logical_and(high_uvalue_indices, dt_labels[cls] == 1)
            filtered_uncertain_voxels = np.logical_or(filtered_uncertain_voxels, correct_voxels)
        return filtered_uncertain_voxels

    def _set_selective_voxels(self, pred_labels, gt_labels, high_uvalue_indices):
        """
        Correct all voxels indicated as highly uncertain with the ground truth label.

        :param pred_labels: [num_of_classes, w, h]
        :param gt_labels: [w, h] multiclass labels
        :param high_uvalue_indices: [w, h]
        :param dt_labels: binary numpy array [nclasses, w, h] indicating segmentation inaccuracies we would like to detect
        :return: correct pred_labels w.r.t. high uncertain voxels
        """
        num_classes = pred_labels.shape[0]
        for cls in np.arange(num_classes):
            pred_labels[cls, high_uvalue_indices] = gt_labels[high_uvalue_indices] == cls

        return pred_labels

    def save(self, network="default"):
        cropped = "_cropped" if self.use_cropped else ""
        detection = "_" + self.dt_config_id if self.dt_config_id is not None else ""
        fname = "cov_risk_" + self.type_of_map + "_" + self.cardiac_phase + cropped + detection + ".npz"
        fname = os.path.join(self.src_data_path, fname)
        np.savez(fname, coverages=self.x_coverages, mean_errors=self.mean_errors,
                 seg_errors=self.seg_errors,
                 dice=self.patient_dsc, hd=self.patient_hd, cardiac_phase=self.cardiac_phase,
                 mc_dropout=self.mc_dropout, use_cropped=1 if self.use_cropped else 0)
        print("INFO - Saved results to {}".format(fname))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate selective classification data to produce risk-coverage'
                                                 'curves')
    parser.add_argument('--seg_model', type=str, help='directory for experiment outputs')
    parser.add_argument('--loss_function', type=str, default="ce")
    parser.add_argument('--channels', type=str, choices=['bmap', 'emap', None], default=None)

    args = parser.parse_args()
    if args.channels is None:
        eval_maps = [False, True]
    else:
        eval_maps = [True if args.channels == "bmap" else False]

    use_cropped = True
    verbose = False
    dt_config_id = None  # "fixed_42_28"
    phases = ['ES', 'ED']
    patients = None # ['patient001', 'patient016',  'patient021', 'patient070', 'patient097']

    exper_dir = os.path.join("~/expers/redo_expers/", args.seg_model + "_" + args.loss_function)
    exper_dir = os.path.expanduser(exper_dir)
    print("INFO - Model {}".format(args.seg_model + "_" + args.loss_function))
    for cardiac_phase in phases:
        for mc_dropout in eval_maps:
            print("INFO - Processing phase/mc-dropout {}/{}".format(cardiac_phase, mc_dropout))
            generator = SelectiveClassification(exper_dir, cardiac_phase, num_of_thresholds=100,
                                                patients=patients, verbose=verbose, mc_dropout=mc_dropout,
                                                dt_config_id=dt_config_id)
            generator.generate(use_cropped=use_cropped)
            generator.save(network=args.seg_model)
