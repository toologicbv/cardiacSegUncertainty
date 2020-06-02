import numpy as np

from utils.medpy_metrics import hd, dc, hd95, assd
from utils.post_processing import filter_connected_components, getLargestCC


class VolumeEvaluation(object):

    def __init__(self, patient_id, pred_labels, ref_labels, num_of_cardiac_phases=1, num_of_classes=4,
                 voxel_spacing=None, mc_dropout=False, cardiac_phase=None, ignore_labels=None):
        """

        :param patient_id:
        :param pred_labels:
        :param ref_labels:
        :param num_of_cardiac_phases:
        :param num_of_classes:
        :param voxel_spacing:
        :param mc_dropout:
        :param cardiac_phase:
        :param ignore_labels: if not None binary vector of length n_classes.
        """
        self.patient_id = patient_id
        self.num_of_classes = num_of_classes
        self.num_of_phases = num_of_cardiac_phases
        self.cardiac_phase = cardiac_phase
        """
            (1) self.ref_label contains the ground truth: [z, y, x]
            (2) self.pred_labels contains predicted labels [z, y, x]
        """
        self.ref_labels = ref_labels
        self.pred_labels = pred_labels
        self.voxel_spacing = voxel_spacing  # IMPORTANT: has shape [z, y, x].
        self.dice = None
        self.hd = None
        self.hd95 = None
        self.assd = None
        self.seg_errors = None
        self.dice_slices = None
        self.hd_slices = None
        self.mc_dropout = mc_dropout
        self.class_indices = list(np.arange(1, self.num_of_classes))
        # remember: ignore_labels is binary vector and with np.nonzero we translate this to a multi-class vector
        if ignore_labels is not None and np.count_nonzero(ignore_labels) != 0:
            # remove the class indices that are not contained in this volume. np.nonzero returns tuple and we only
            # need the first returned array because ignore_labels is a vector 1D)
            for cls_idx in np.nonzero(ignore_labels)[0]:
                self.class_indices.remove(cls_idx)

    def get_seg_errors(self):
        seg_errors = np.zeros_like(self.pred_labels)
        for cls_idx in self.class_indices:
            seg_errors[(self.pred_labels == cls_idx) != (self.ref_labels == cls_idx)] = 1

        return seg_errors

    def fast_evaluate(self, compute_hd=False, apply_post_processing=True):
        """
        Remember the shape of self.ref_labels is [z, y, x] => multi-labels whereas the shape of self.pred_labels
        is [z, nclasses, y, x]

        :param compute_hd:
        :param apply_post_processing:
        :return:
        """
        self.dice = np.zeros(self.num_of_phases * self.num_of_classes)
        self.hd = np.zeros(self.num_of_phases * self.num_of_classes)
        self.hd95 = np.zeros(self.num_of_phases * self.num_of_classes)
        self.assd = np.zeros(self.num_of_phases * self.num_of_classes)
        new_segmentation = np.zeros_like(self.pred_labels)

        for cls_idx in self.class_indices:
            if np.count_nonzero(self.pred_labels == cls_idx) != 0:
                if apply_post_processing:
                    new_segmentation[getLargestCC(self.pred_labels == cls_idx) == 1] = cls_idx
                else:
                    new_segmentation[self.pred_labels == cls_idx] = cls_idx
                if np.count_nonzero(self.ref_labels == cls_idx) != 0:
                    self.dice[cls_idx] = dc(new_segmentation == cls_idx, getLargestCC(self.ref_labels == cls_idx))

        if compute_hd:
            # we use filtered auto segmentations to compute HD
            for cls_idx in self.class_indices:
                if np.count_nonzero(new_segmentation == cls_idx) != 0 and np.count_nonzero(self.ref_labels == cls_idx) != 0:
                    self.hd[cls_idx] = hd(new_segmentation == cls_idx, getLargestCC(self.ref_labels == cls_idx),
                                          self.voxel_spacing)
                    self.hd95[cls_idx] = hd95(new_segmentation == cls_idx, getLargestCC(self.ref_labels == cls_idx),
                                              self.voxel_spacing)
                    self.assd[cls_idx] = assd(new_segmentation == cls_idx, getLargestCC(self.ref_labels == cls_idx),
                                              self.voxel_spacing)
        # dcs = list()
        # dcs.append([dc(new_segmentation == lab, getLargestCC(self.ref_labels == lab)) for lab in self.class_indices])
        # dcs = np.array(dcs)[0]
        # for r_idx, cls_idx in enumerate(self.class_indices):
        #     self.dice[cls_idx] = dcs[r_idx]
        self.pred_labels = new_segmentation

    def post_processing_only(self):

        new_segmentation = np.zeros_like(self.pred_labels)
        for cls_idx in self.class_indices:
            if np.count_nonzero(self.pred_labels == cls_idx) != 0:
                new_segmentation[getLargestCC(self.pred_labels == cls_idx) == 1] = cls_idx
        self.pred_labels = new_segmentation

    def show_results(self):

        print("Image {} - {}"
               " dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
               "".format(self.patient_id, self.cardiac_phase,
                                                self.dice[1], self.dice[2],
                                                self.dice[3]))
        if self.hd is not None:
            print("\t\t\t\t\t"
                   "Hausdorff(RV/Myo/LV):\t{} {:.2f}/{:.2f}/{:.2f}\t"
                   "".format(self.cardiac_phase, self.hd[1], self.hd[2],
                                                    self.hd[3]))

