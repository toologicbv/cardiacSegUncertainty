import numpy as np
import os
import glob
import traceback


def compute_cross_fold_results(result_dice, result_hd, do_show=False):
    """

    :param result_dice: dictionary, key fold_id, contains np array of shape [#patients, 2, 4]
    :param result_hd: dictionary, key fold_id, contains np array of shape [#patients, 2, 4]
    Compute the overall result for this
    :return: results averaged over all patients (in the 4 folds)

    """
    dice, hd = np.empty((0, 2, 4)), np.empty((0, 2, 4))
    for fold_id, f_dice in result_dice.items():
        f_hd = result_hd[fold_id]
        dice = np.vstack([dice, f_dice]) if dice.size else f_dice
        hd = np.vstack([hd, f_hd]) if hd.size else f_hd

    mean_dice = np.mean(dice, axis=0)
    mean_hd = np.mean(hd, axis=0)

    if do_show:
        print(("dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
              "ED {:.2f}/{:.2f}/{:.2f}".format(mean_dice[0, 1], mean_dice[0, 2],
                                               mean_dice[0, 3], mean_dice[1, 1],
                                               mean_dice[1, 2], mean_dice[1, 3])))
        print(("Hausdorff(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
              "ED {:.2f}/{:.2f}/{:.2f}".format(mean_hd[0, 1], mean_hd[1, 2],
                                               mean_hd[0, 3], mean_hd[1, 1],
                                               mean_hd[1, 2], mean_hd[1, 3])))

    return mean_dice, mean_hd


class FoldEvaluation(object):

    file_extension = "seg_evaluation"
    default_size = 25

    def __init__(self, model_name, fold, result_root_dir='~/expers/dcnn', type_of_map="e_map"):

        """

        :param model_name:
        :param fold:
        :param result_root_dir: directory where we store the results for this evaluation
                                important: should identify the model we're evaluating
        :param type_of_map:
        """
        if type_of_map not in ["b_map", "e_map"]:
            raise ValueError("ERROR - type_of_maps must be e_map or u_map")
        self.fold_id = fold
        self.model_name = model_name
        # IMPORTANT: patient_ids is a numpy array, so we don't store "patient012" but "12".
        # FoldEvaluation has to static methods to convert the int IDs back to string IDs (see below)
        # BUT!!!
        # The object trans_dict_slices is a dictionary with key PATIENT_ID
        self.patient_ids = np.empty(0)
        self.dice = np.empty(0)
        self.hd = np.empty(0)
        self.dice_slices = np.empty(0)
        self.hd_slices = np.empty(0)
        self.seg_errors = np.empty(0)
        # construct output dir
        self.model_base_dir = os.path.expanduser(result_root_dir)
        self.output_dir_results = None
        self.type_of_map = type_of_map
        if type_of_map == "b_maps":
            self.mc_dropout = True
        else:
            self.mc_dropout = False
        self._check_dirs()
        # translation of patient_id (key) to numpy index in array self.hd, self.dice (and same for slices)
        self.trans_dict = {}
        self.trans_dict_slices = {}
        # dictionaries that hold numpy 1D vectors for mpl boxplot visualization of results
        self.bp_list_hd = None
        self.bp_list_dice = None

    def _check_dirs(self):
        if self.output_dir_results is None:
            self.output_dir_results = os.path.join(self.model_base_dir, "results")
            if not os.path.isdir(self.output_dir_results):
                os.mkdir(self.output_dir_results)
            # self.output_dir_results = os.path.join(self.output_dir_results, self.model_name)
            # if not os.path.isdir(self.output_dir_results):
            #     os.mkdir(self.output_dir_results)
            # self.output_dir_results = os.path.join(self.output_dir_results, self.type_of_map)
            # if not os.path.isdir(self.output_dir_results):
            #     os.mkdir(self.output_dir_results)

    def add_patient_eval(self, p_eval_obj):
        """

        :param p_eval_obj: VolumeEvaluation object
        :return:
        """
        dice = np.reshape(p_eval_obj.dice, (1, 2, 4))
        self.dice = np.vstack([self.dice, dice]) if self.dice.size else dice

        hd = np.reshape(p_eval_obj.hd, (1, 2, 4))
        self.hd = np.vstack([self.hd, hd]) if self.hd.size else hd
        # sum over slice axis

        if p_eval_obj.seg_errors is not None:
            # seg_errors is numpy array of shape [#classes, #slices]
            seg_errors = np.expand_dims(np.sum(p_eval_obj.seg_errors, axis=1), axis=0)
            self.seg_errors = np.vstack([self.seg_errors, seg_errors]) if self.seg_errors.size else seg_errors

        p_id = FoldEvaluation.patient_id_to_int(p_eval_obj.patient_id)
        self.patient_ids = np.vstack([self.patient_ids, p_id]) if self.patient_ids.size else np.array(p_id)

        self.trans_dict[p_eval_obj.patient_id] = self.dice.shape[0] - 1

        # if not none process slice performance measures
        if p_eval_obj.dice_slices is not None:
            # we store the offset index for each patient that can be used to extract the dice and hd values
            # for each slice of the patient. e.g. self.hd_slices has shape [#total_slices, 2, 4]
            # so #total_slices contains all slices not just for one patient. We can use the index
            # to extract the corresponding slices for this patient, see method get_patient_slice_measures
            self.trans_dict_slices[p_eval_obj.patient_id] = self.dice_slices.shape[0]
            dice_slices = np.swapaxes(p_eval_obj.dice_slices, 0, 1)
            dice_slices = np.reshape(dice_slices, (dice_slices.shape[0], 2, 4))
            self.dice_slices = np.vstack([self.dice_slices, dice_slices]) if self.dice_slices.size else dice_slices

        if p_eval_obj.hd_slices is not None:
            hd_slices = np.swapaxes(p_eval_obj.hd_slices, 0, 1)
            hd_slices = np.reshape(hd_slices, (hd_slices.shape[0], 2, 4))
            self.hd_slices = np.vstack([self.hd_slices, hd_slices]) if self.hd_slices.size else hd_slices

    def get_patient_slice_measures(self, patient_id):
        pat_seq_nbr = self.trans_dict[patient_id]
        start_idx = list(self.trans_dict_slices.values())[pat_seq_nbr]
        if len(list(self.trans_dict_slices.keys())) == pat_seq_nbr + 1:
            end_idx = None
        else:
            end_idx = list(self.trans_dict_slices.values())[pat_seq_nbr + 1]
        # print("{} {} {}".format(pat_seq_nbr, start_idx, end_idx))
        return self.dice_slices[start_idx:end_idx], self.hd_slices[start_idx:end_idx]

    def generate_box_plot_input(self, per_phase=True, per_class=True):

        self.bp_list_dice = {"ES": {'RV': [], 'MYO': [], 'LV': []}, "ED": {'RV': [], 'MYO': [], 'LV': []}}
        self.bp_list_hd = {"ES": {'RV': [], 'MYO': [], 'LV': []}, "ED": {'RV': [], 'MYO': [], 'LV': []}}
        # self.dice has shape [#patient, 2, 4] 0=ES, 1=ED; 0=BG, 1=RV, 2=MYO, 3=LV
        self.bp_list_dice["ES"]["RV"] = self.dice[:, 0, 1]
        self.bp_list_dice["ES"]["MYO"] = self.dice[:, 0, 2]
        self.bp_list_dice["ES"]["LV"] = self.dice[:, 0, 3]
        self.bp_list_dice["ED"]["RV"] = self.dice[:, 1, 1]
        self.bp_list_dice["ED"]["MYO"] = self.dice[:, 1, 2]
        self.bp_list_dice["ED"]["LV"] = self.dice[:, 1, 3]

        # same for HD
        self.bp_list_hd["ES"]["RV"] = self.hd[:, 0, 1]
        self.bp_list_hd["ES"]["MYO"] = self.hd[:, 0, 2]
        self.bp_list_hd["ES"]["LV"] = self.hd[:, 0, 3]
        self.bp_list_hd["ED"]["RV"] = self.hd[:, 1, 1]
        self.bp_list_hd["ED"]["MYO"] = self.hd[:, 1, 2]
        self.bp_list_hd["ED"]["LV"] = self.hd[:, 1, 3]

    def get_boxplot_vectors(self, force_reload=True):
        if force_reload or self.bp_list_dice is None:
            self.generate_box_plot_input()
        return self.bp_list_dice, self.bp_list_hd

    def save(self, file_name=None):
        if self.patient_ids.size:
            self.patient_ids = np.squeeze(self.patient_ids)

        if file_name is None:
            file_name = "{}_f{}_".format(self.dice.shape[0], self.fold) + FoldEvaluation.file_extension

        abs_file_name = os.path.join(self.output_dir_results, file_name)

        try:
            np.savez(abs_file_name, dice=self.dice, hd=self.hd, seg_errors=self.seg_errors,
                     patient_ids=self.patient_ids, dice_slices=self.dice_slices, hd_slices=self.hd_slices,
                     pat_slice_indices=np.array(list(self.trans_dict_slices.values())))
            print(("INFO - Successfully saved eval-results to {}".format(abs_file_name)))
        except IOError:
            print(("ERROR - Unable to save results to file {}".format(file_name)))
            raise
        except Exception as e:
            print((traceback.format_exc()))
            raise

    def load(self, file_name=None):
        if file_name is None:
            search_mask = "*" + FoldEvaluation.file_extension + ".npz"
        else:
            search_mask = file_name
            if search_mask.find(".npz") == -1:
                search_mask += ".npz"

        search_path = os.path.join(self.output_dir_results, search_mask)
        file_list = glob.glob(search_path)
        if len(file_list) > 1:
            print("WARNING - No file found or more than than one file to load")
            for fname in file_list:
                print(fname)
            print("Please specify by means of file_name parameter (no absolute path required)")
            raise ValueError("ERROR - More than one file to load")
        elif len(file_list) == 0:
            raise IOError("ERROR - no files found for {}".format(search_path))

        file_to_load = file_list[0]
        try:
            data = np.load(file_to_load)
            self.dice = data["dice"]
            self.hd = data["hd"]
            self.seg_errors = data["seg_errors"]
            self.patient_ids = data["patient_ids"]
            if "pat_slice_indices" in list(data.keys()):
                pat_slice_indices = data["pat_slice_indices"]
                self.dice_slices = data["dice_slices"]
                self.hd_slices = data["hd_slices"]
            else:
                pat_slice_indices = None
            for idx, p_id in enumerate(self.patient_ids):
                patient_id = FoldEvaluation.int_to_patient_id(p_id)
                self.trans_dict[patient_id] = idx
                if pat_slice_indices is not None:
                    self.trans_dict_slices[patient_id] = pat_slice_indices[idx]

            print(("INFO - Successfully loaded numpy archives from {}".format(file_to_load)))
        except Exception as e:
            print((traceback.format_exc()))
            raise


