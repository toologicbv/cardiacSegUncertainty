import os
import glob
import numpy as np
from collections import OrderedDict
import copy
from tqdm import tqdm
from utils.common import loadExperimentSettings

from utils.detection.heat_map import ImageHeatMapHandler
from evaluate.acdc_patient_eval import VolumeEvaluation
from utils.medpy_metrics import dc
from evaluate.test_results import ACDCTestResult
from datasets.ACDC.data import get_acdc_patient_ids
from datasets.ACDC.get_data import load_data
from datasets.common import get_config
from evaluate.common import save_nifty

dta_settings = get_config('ACDC')


def collapse_roi_maps(target_roi_maps):
    """
    our binary roi maps specify the voxels to "inspect" per tissue class, for training this distinction
    doesn't matter, so here, we collapse the classes to 1
    :param target_roi_maps: [#slices, #classes, w, h]
    :return: [#slices, w, h]

    """
    target_roi_maps = np.sum(target_roi_maps, axis=1)
    # kind of logical_or over all classes. if one voxels equal to 1 or more, than set voxel to 1 which means
    # we need to correct/inspect that voxel
    target_roi_maps[target_roi_maps >= 1] = 1
    return target_roi_maps.astype(np.int)


class SimulateExpert(object):
    nclasses = 4

    key_save_arrays = ["dice_es", "dice_ed", "hd_es", "hd_ed", "dice_es_std", "surf_dice_es", "surf_dice_ed",
                       "dice_ed_std", "hd_es_std", "hd_ed_std", "pat_frame_ids",
                       "seg_error_det_es", "seg_error_det_ed", "slice_dice_es", "slice_dice_ed",
                       "slice_hd_es", "slice_hd_ed",
                       "mean_dice_es", "mean_dice_ed", "mean_hd_es", "mean_hd_ed",
                       "std_dice_es", "std_dice_ed", "std_hd_es", "std_hd_ed"]

    def __init__(self, path_detection_dir, patient_ids=None, all_folds_settings=None,
                 correct_all_seg_errors=False, verbose=False, eval_run_id=None):
        """
        -------------------------------------------------------------------------------------
        Example usage:
        patient_ids = None  # ["patient016", "patient017"]
        src_exper_dir = os.path.expanduser("~/expers/acdc/unet_mc_dice")
        all_folds_settings = {'network': 'unet_mc', 'input_channels': 'bmap', }
        # fold0:["patient016","patient017", "patient018", "patient019", "patient020"]
        sim_expert = SimulateExpert(src_exper_dir, all_folds_settings=all_folds_settings,
                                    patient_ids=patient_ids, verbose=True, correct_all_seg_errors=False)
        sim_expert.evaluate_detections(evaluate_baseline=True)
        sim_expert.save()
        -------------------------------------------------------------------------------------
        :param path_detection_dir: absolute path to dir that stores details about detection model training
        :param all_folds_settings: dictionary that should be not NONE in case path_detection_dir
                    does not point to an experiment dir with settings.yaml file.
                    In this case we're processing ALL FOLDS and we need to pass the settings:
                    {'input_channels': emap/bmap, 'dt_config_id': 'fixed_46_31'
                     'network': drn_mc, dcnn_mc, unet_mc, }
        :param patient_ids:
        :param correct_all_seg_errors: boolean, IMPORTANT: if set to TRUE we will correct ALL filtered segmentation
                                       errors (t_roi_maps) without taking the region detection heat maps into account
                                       This is used as a kind of BASELINE.
        :param dt_config_id: extension for directories self.dir_dt_maps and self.dir_detector_labels in order to
                             separate different configurations for error detection
        :param eval_run_id: in case we want to run different evaluations for the same dt_config_id we specify a separate
                            eval_run_id. NOTE: you also need to specify this for the EvaluationHandler when creating the
                            actual heat maps.
        :param verbose:
        """
        self.patient_ids = patient_ids
        self.correct_all_seg_errors = correct_all_seg_errors
        if os.path.isfile(os.path.join(path_detection_dir, 'settings.yaml')):
            self.exper_settings = loadExperimentSettings(os.path.join(path_detection_dir, 'settings.yaml'))
            self._settings()
        else:
            # path_detection_dir is actually not an exper dir under .../dt_logs. Meaning, we're not processing a
            # specific detection model/fold but just ALL patient ids we can find in the HEAT_MAP_DIR
            # So option is useful if we want to evaluate ALL folds in one go.
            # path_detection_dir should be something like: ~/expers/acdc/dcnn_mc_brier/
            #   We assume there exists a "heat_maps" directory under this main dir
            assert all_folds_settings is not None
            self._no_settings(path_detection_dir, all_folds_settings)

        assert self.input_channels in ['bmap', 'emap']
        print("INFO - Working with settings model: {} fold: {} dropout: {} "
              "dt_config: {} input-channels: {}".format(self.model_name, self.fold, self.mc_dropout,
                                                                                           self.dt_config_id,
                                                                                           self.input_channels))
        self.eval_run_id = eval_run_id
        self.verbose = verbose

        self.num_cardiac_phases = 1
        self.heat_map_dir = None
        self.corr_pred_labels_dir = None
        self.output_dir_results = None
        self.result_dir = None
        self.heat_maps = OrderedDict()
        self.corrected_pred_labels = OrderedDict()
        self.detected_voxel_mask = OrderedDict()
        self.tp_detection_rate = OrderedDict()
        self.patient_frame_array = np.empty(0)
        self.patient_frame_array = np.empty(0)

        self._check_dirs()
        self.data_handler = None
        self.heat_map_handler = None

        # keep the evaluation objects per patient
        self.base_fold_eval_obj = None
        self.fold_eval_obj = None
        # important: _check_dirs has to be called first
        self._get_patient_ids()

        print("Warning - Using heat maps: {} ".format(not self.correct_all_seg_errors))
        # if not correct_all_seg_errors:
        #    print("WARNING - You're reporting detection performance by treating ALL seg-errors in patch as"
        #          "corrected (not only the filtered-seg-errors")

    def _get_patient_ids(self):
        if self.patient_ids is None:
            if self.fold is not None:
                pat_ids = get_acdc_patient_ids(self.fold, "validation", limited_load=False)
                self.patient_ids = ["patient{:03d}".format(patid) for patid in pat_ids]
            else:
                # processing all folds
                if self.correct_all_seg_errors:
                    # we replicate segmentation results for all folds
                    self.patient_ids = ["patient{:03d}".format(patid) for patid in np.arange(1, 101)]
                else:
                    # only process the patient ids you can find in the heat map directory. Make sure
                    # that _check_dirs() has been called before in __init__
                    f_mask = "*_mc.nii.gz" if self.mc_dropout else "*E?.nii.gz"
                    search_mask = os.path.join(self.heat_map_dir, f_mask)
                    flist = glob.glob(search_mask)
                    self.patient_ids = set([os.path.basename(fname).split("_")[0] for fname in flist])
            if len(self.patient_ids) == 0:
                raise ValueError("ERROR - no heat maps found in {}".format(search_mask))
        print("self.patient_ids 2 ", self.patient_ids)
        print("# patients to process {}".format(len(self.patient_ids)))

    def _no_settings(self, src_path_data, all_folds_settings):
        self.src_data_path = src_path_data
        self.abs_path_fold = self.src_data_path  # same as src_data_path, we're processing ALL FOLDS
        self.dt_config_id = "fixed_" + str(dta_settings.dt_margins[0]).replace(".", "") + "_" +\
                            str(dta_settings.dt_margins[1]).replace(".", "")
        self.input_channels = all_folds_settings['input_channels']
        self.model_name = all_folds_settings['network']
        self.fold = None
        self.mc_dropout = True if self.input_channels == "bmap" else False

    def _settings(self):
        # e.g. ~/expers/acdc/drn_mc_ce/ which should hold for all patients the predicted labels, umaps and detection
        # labels
        self.src_data_path = self.exper_settings.src_path_data
        self.dt_config_id = self.exper_settings.dt_config_id
        self.type_of_map = 'bmap' if self.exper_settings.mc_dropout else 'emap'
        self.model_name = self.exper_settings.network
        self.fold = self.exper_settings.fold
        self.mc_dropout = self.exper_settings.mc_dropout

        if "input_channels" not in vars(self.exper_settings).keys():
            self.input_channels = "allchannels"
        else:
            self.input_channels = self.exper_settings.input_channels

        # absolute path to the directory in which training details of detection model resides.
        # e.g. ~/expers/acdc/dcnn_mc_brier/dt_logs/f0_rsn_065_008_20190725_173416/
        self.abs_path_fold = self.exper_settings.output_directory

    def _check_dirs(self):
        if self.heat_map_dir is None:
            self.heat_map_dir = os.path.join(self.abs_path_fold, "heat_maps")
            self.corr_all_pred_labels_dir = self.heat_map_dir.replace("heat_maps", "corr_all_pred_labels")
            self.corr_pred_labels_dir = self.heat_map_dir.replace("heat_maps", "corr_pred_labels")
            print("WARNING - Load heat maps from {}".format(self.heat_map_dir))

        if self.result_dir is None:
            self.result_dir = os.path.join(self.abs_path_fold, "sim_results" + os.sep + self.input_channels)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=False)

    def _check_dict(self, mydict, patient_id):
        if patient_id not in mydict.keys():
            mydict[patient_id] = {}

    def _get_pat_data(self, patient_id, cardiac_phase):
        data_to_load = ["pred_labels", "dt_labels", "umaps", "ref_labels"]
        data_dict = load_data(self.src_data_path, cardiac_phase, data_to_load,
                              mc_dropout=self.mc_dropout, dt_config_id=self.dt_config_id,
                              patient_id=patient_id, one_hot=False)
        data_handler = {d: data_dict[d][patient_id] for d in data_to_load}
        data_handler['voxel_spacing'] = data_dict['umaps'][patient_id]['spacing']
        # Load heat maps
        if not self.correct_all_seg_errors:
            heat_map_handler = ImageHeatMapHandler(mode_dataset="separate")
            heat_map_handler.load_heat_map(self.heat_map_dir, patient_id, cardiac_phase, mc_dropout=self.mc_dropout)
            data_handler['heat_map'] = heat_map_handler.heat_maps[patient_id][cardiac_phase]
        return data_handler

    def evaluate_detections(self, evaluate_baseline=False, save_new_labels=False):

        fold_eval_obj = ACDCTestResult()
        if evaluate_baseline:
            self.base_fold_eval_obj = ACDCTestResult()

        for p_id in tqdm(sorted(self.patient_ids), desc="Evaluate with detection"):
            for cardiac_phase in ['ED', 'ES']:
                self.data_handler = self._get_pat_data(p_id, cardiac_phase)
                voxel_spacing = self.data_handler['voxel_spacing']
                num_slices, w, h = self.data_handler['pred_labels'].shape
                self._determine_voxels(p_id, cardiac_phase, prob_theta=SimulateExpert._determine_threshold())
                # 3-tuple: patid (integer), cardiac_phase string, #slices
                pat_frame_id = np.array([int(p_id.strip("patient")), cardiac_phase, int(num_slices)])
                self.patient_frame_array = np.vstack((self.patient_frame_array, pat_frame_id)) \
                    if self.patient_frame_array.size else pat_frame_id
                eval_obj = VolumeEvaluation(p_id, self.data_handler['corrected_pred_labels'],
                                            self.data_handler['ref_labels'],
                                            voxel_spacing=voxel_spacing, num_of_classes=4,
                                            mc_dropout=self.mc_dropout, cardiac_phase=cardiac_phase)
                eval_obj.fast_evaluate(compute_hd=True, apply_post_processing=False)
                fold_eval_obj(eval_obj.dice, hd=eval_obj.hd, cardiac_phase_tag=cardiac_phase, pat_id=p_id,
                              hd95=eval_obj.hd95, assd=eval_obj.assd
                              )
                # still_errors = ref_labels[1:] != eval_obj.pred_labels[1:]
                # print("Patient {}/{}".format(p_id, frame_id), eval_obj.dice)
                # print("Before ", np.sum(before_errors, axis=(1, 2, 3)))
                # print("After  ", np.sum(still_errors, axis=(1, 2, 3)))
                # t = np.sum(before_errors, axis=(1, 2, 3)) - np.sum(still_errors, axis=(1, 2, 3))
                # print("Diff {}".format(t))
                if save_new_labels:
                    self.save_corrected_labels(p_id, cardiac_phase)
                if evaluate_baseline:
                    self.evaluate_dcnn_mc_base_model(p_id, cardiac_phase, voxel_spacing)

        self.fold_eval_obj = fold_eval_obj
        print("Dice ")
        print(self.fold_eval_obj.get_stats_dice()['mean'])
        if evaluate_baseline:
            print("Baseline DSC")
            print(self.base_fold_eval_obj.get_stats_dice()['mean'])
        print("HD ")
        print(self.fold_eval_obj.get_stats_hd()['mean'])
        if evaluate_baseline:
            print("Baseline HD")
            print(self.base_fold_eval_obj.get_stats_hd()['mean'])

        if self.fold is None:
            # we save the overall results for all patients as test result object to file
            res_suffix = "_mc.npz" if self.mc_dropout else ".npz"
            res_prefix = "results_fall_corrall" if self.correct_all_seg_errors else "results_fall"
            fname = os.path.join(self.src_data_path,
                                 res_prefix + "_{}".format(len(self.fold_eval_obj.pat_ids)) +
                                 res_suffix)
            self.fold_eval_obj.save(filename=fname)
            if evaluate_baseline:
                fname = os.path.join(self.src_data_path,
                                       "results_base_fall" + "_{}".format(len(self.fold_eval_obj.pat_ids)) +
                                     res_suffix)
                self.base_fold_eval_obj.save(filename=fname)
        del fold_eval_obj

    def evaluate_dcnn_mc_base_model(self, patient_id, cardiac_phase, voxel_spacing):

        pat_eval = VolumeEvaluation(patient_id, self.data_handler['pred_labels'], self.data_handler['ref_labels'],
                                    voxel_spacing=voxel_spacing, num_of_classes=4,
                                    mc_dropout=self.mc_dropout, cardiac_phase=cardiac_phase)
        pat_eval.fast_evaluate(compute_hd=True, apply_post_processing=False)
        self.base_fold_eval_obj(pat_eval.dice, hd=pat_eval.hd, cardiac_phase_tag=cardiac_phase, pat_id=patient_id,
                                hd95=pat_eval.hd95, assd=pat_eval.assd)
        # print("Patient {} {}".format(patient_id, frame_id))
        # print(eval_obj.dice)

    def _determine_voxels(self, patient_id, cardiac_phase, prob_theta=0.5):

        # Step 1: generate a mask that contains the voxels to be corrected based on heat map
        #         returns mask of shape [#slices, w, h] for one cardiac phase
        self._check_dict(self.corrected_pred_labels, patient_id)
        voxels_mask_to_be_corrected = self._get_voxels_to_be_corrected(patient_id, cardiac_phase, prob_theta=prob_theta)
        self.data_handler['detected_voxel_mask'] = voxels_mask_to_be_corrected
        self.data_handler['corrected_pred_labels'] = \
                copy.deepcopy(self.data_handler['pred_labels'])
        if np.count_nonzero(voxels_mask_to_be_corrected) != 0:
            if self.verbose:
                print(("INFO - Generate new predicted labels for patient {}".format(patient_id)))
            voxels_mask_to_be_corrected = np.atleast_1d(voxels_mask_to_be_corrected.astype(np.bool))
            self.data_handler['corrected_pred_labels'][voxels_mask_to_be_corrected] = \
                self.data_handler['ref_labels'][voxels_mask_to_be_corrected]

    def _get_voxels_to_be_corrected(self, p_id, cardiac_phase, prob_theta):
        self._check_dict(self.tp_detection_rate, patient_id=p_id)
        # remember target roi's have shape [slices, w, h]
        # and heat maps [ #slices, w, h].
        target_seg_errors = np.atleast_1d(self.data_handler['dt_labels'].astype(np.bool))
        target_seg_errors = collapse_roi_maps(target_seg_errors)
        # target_seg_errors has [z, y, x] shape. Count errors per slice
        errors_to_be_detected = np.count_nonzero(target_seg_errors, axis=(1, 2)).astype(np.float)

        if np.count_nonzero(target_seg_errors) != 0:
            if not self.correct_all_seg_errors:
                heat_map = self.data_handler['heat_map']
                heat_map_indices = heat_map >= prob_theta

            ref_seg_errors_indices = target_seg_errors == 1
            if self.correct_all_seg_errors:
                voxels_mask_to_be_corrected = ref_seg_errors_indices
            else:
                voxels_mask_to_be_corrected = np.logical_and(heat_map_indices, ref_seg_errors_indices)

            errors_detected = np.count_nonzero(voxels_mask_to_be_corrected, axis=(1, 2)).astype(np.float)
            self.tp_detection_rate[p_id][cardiac_phase] = np.concatenate((errors_detected[np.newaxis],
                                                                     errors_to_be_detected[np.newaxis]))
            if self.verbose:
                print(("INFO - patient {}: {}".format(p_id, np.count_nonzero(voxels_mask_to_be_corrected))))

        else:
            voxels_mask_to_be_corrected = np.zeros_like(target_seg_errors)
            self.tp_detection_rate[p_id][cardiac_phase] = np.concatenate((np.zeros_like(errors_to_be_detected[np.newaxis]),
                                                                     errors_to_be_detected[np.newaxis]))
            if self.verbose:
                print(("INFO - no target seg-errors found for patient id {}".format(p_id)))

        # print("Patient {}/{} #total-target {} #detected {}".format(p_id, cardiac_phase, np.sum(errors_to_be_detected),
        #                                                           np.sum(errors_detected)))
        return voxels_mask_to_be_corrected

    def _compress_detection_stats(self):
        # Remember self.tp_detection_rate is Dict of Dict, Patient, frame id and contains
        # for each pat/frame numpy array of [2, #slices], 0=detected errors per slice, 1=to be detected errors per slice
        # Here we convert the dict/dict to a numpy array and sum over the slices (per patient). Which leaves us with two
        # stats per pat/frame id 0=total #errors detected 1=total #errors to be detected
        tp_detection_es = np.zeros((len(self.patient_ids), 2))
        tp_detection_ed = np.zeros((len(self.patient_ids), 2))
        i_es, i_ed = 0, 0
        for p_id, cardiac_phase, num_slices in self.patient_frame_array:
            patient_id = "patient{:03d}".format(int(p_id))
            # sum over slices (axis=1)
            dt_pat = np.sum(self.tp_detection_rate[patient_id][cardiac_phase], axis=1)
            if cardiac_phase == "ES":
                tp_detection_es[i_es] = dt_pat
                i_es += 1
            else:
                tp_detection_ed[i_ed] = dt_pat
                i_ed += 1

        return {"ES": tp_detection_es, "ED": tp_detection_ed}

    @staticmethod
    def _determine_threshold():
        # TODO: Needs implementation. This is the threshold used for the region heat maps to determine
        # which voxels will be corrected by the simulated human-in-the-loop
        return 0.5

    def save(self):
        """
        ["dice_es", "dice_ed", "hd_es", "hd_ed", "dice_es_std",
                       "dice_ed_std", "hd_es_std", "hd_ed_std", "pat_frame_ids",
                       "seg_error_det_es", "seg_error_det_ed"]
        :param:
        :return:
        """
        if self.correct_all_seg_errors:
            eval_tag = "sim_expert_allerrors_"
        else:
            eval_tag = "sim_expert_"
        save_arrays = {k: None for k in SimulateExpert.key_save_arrays}
        save_arrays["pat_frame_ids"] = self.patient_frame_array
        num_evals = self.patient_frame_array.shape[0]

        # is ACDCTestResult object from utils.acdc.helpers
        save_arrays["dice_es"], save_arrays["dice_ed"] = self.fold_eval_obj.dice["ES"], self.fold_eval_obj.dice["ED"]

        save_arrays["hd_es"], save_arrays["hd_ed"] = self.fold_eval_obj.hd["ES"], self.fold_eval_obj.hd["ED"]
        tp_detection = self._compress_detection_stats()
        save_arrays["seg_error_det_es"] = tp_detection["ES"]
        save_arrays["seg_error_det_ed"] = tp_detection["ED"]
        extra_tag = "" if self.correct_all_seg_errors else "_" + self.input_channels
        file_name = eval_tag + self.dt_config_id + extra_tag + \
                    "_f{}_n{}.npz".format(self.fold if self.fold is not None else "all", num_evals)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=False)
        abs_file_name = os.path.join(self.result_dir, file_name)
        np.savez(abs_file_name, **save_arrays)
        print("INFO - Saved results to {}".format(abs_file_name))
        # we're only saving the base line evaluation if we're not running the evaluation in which all seg-errors are
        # corrected without taking heat maps into account
        if self.base_fold_eval_obj is not None:  # and not self.correct_all_seg_errors:
            self._save_base_performance(eval_tag.replace("_allerrors", ""))

    def _save_base_performance(self, eval_tag):

        save_arrays = {k: None for k in SimulateExpert.key_save_arrays}
        save_arrays["pat_frame_ids"] = self.patient_frame_array
        num_evals = self.patient_frame_array.shape[0]
        save_arrays["dice_es"], save_arrays["dice_ed"] = self.base_fold_eval_obj.dice["ES"], self.base_fold_eval_obj.dice["ED"]
        save_arrays["hd_es"], save_arrays["hd_ed"] = self.base_fold_eval_obj.hd["ES"], self.base_fold_eval_obj.hd["ED"]
        file_name = eval_tag + "base_f{}_n{}.npz".format(self.fold if self.fold is not None else "all", num_evals)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=False)
        abs_file_name = os.path.join(self.result_dir, file_name)
        np.savez(abs_file_name, **save_arrays)
        print("INFO - Saved (base) results to {}".format(abs_file_name))

    @staticmethod
    def load_results(abs_file_name, as_test_result=False):
        loaded_arrays = {k: None for k in SimulateExpert.key_save_arrays}
        data = np.load(abs_file_name, allow_pickle=True)
        for archive in data.files:
            loaded_arrays[archive] = data[archive]
        print("WARNING - Loading results from {}".format(abs_file_name))
        if as_test_result:
            test_result = ACDCTestResult()
            test_result.cardiac_phases = {'ES': None, 'ED': None}
            test_result.dice['ES'], test_result.dice['ED'] = loaded_arrays['dice_es'], loaded_arrays['dice_ed']
            test_result.hd['ES'], test_result.hd['ED'] = loaded_arrays['hd_es'], loaded_arrays['hd_ed']

            test_result.pat_frame_ids = loaded_arrays['pat_frame_ids']
            return test_result
        else:
            loaded_arrays['mean_dice_ed'], loaded_arrays['std_dice_ed'] = np.mean(loaded_arrays['dice_ed'], axis=0), \
                                                                      np.std(loaded_arrays['dice_ed'], axis=0)
            loaded_arrays['mean_dice_es'], loaded_arrays['std_dice_es'] = np.mean(loaded_arrays['dice_es'], axis=0), \
                                                                      np.std(loaded_arrays['dice_es'], axis=0)
            loaded_arrays['mean_hd_ed'], loaded_arrays['std_hd_ed'] = np.mean(loaded_arrays['hd_ed'], axis=0), \
                                                                      np.std(loaded_arrays['hd_ed'], axis=0)
            loaded_arrays['mean_hd_es'], loaded_arrays['std_hd_es'] = np.mean(loaded_arrays['hd_es'], axis=0), \
                                                                      np.std(loaded_arrays['hd_es'], axis=0)
            return loaded_arrays

    def compare_results(self, patient_ids=None, cardiac_phase=None, tissue_class=None):
        """

        :param patient_ids:
        :param caridac_phase: 0=ES; 1=ED
        :param tissue_class: 0=BG, 1=RV, 2=MYO, 3=LV
        :return:
        """
        if patient_ids is None:
            if self.base_fold_eval_obj is not None:
                patient_ids = list(self.base_fold_eval_obj.trans_dict_slices.keys())
            elif self.fold_eval_obj is not None:
                patient_ids = list(self.fold_eval_obj.trans_dict_slices.keys())
            else:
                raise ValueError("ERROR - base_fold_eval_obj and fold_eval_obj object are both None. "
                                 "At least one must be loaded or generated.")

        if cardiac_phase is None:
            all_tissue_classes = True
        else:
            all_tissue_classes = False

        for p_id in patient_ids:
            if all_tissue_classes:
                np_index_vol = self.base_fold_eval_obj.trans_dict[p_id]
                np_index_slices = self.base_fold_eval_obj.get_patient_slice_measures(patient_id=p_id)
                dice = self.base_fold_eval_obj.dice[np_index_vol]
                print(("------------------------- {} --------------------".format(p_id)))
                print(("before: dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                      "ED {:.2f}/{:.2f}/{:.2f}".format(
                                                       dice[0, 1], dice[0, 2],
                                                       dice[0, 3], dice[1, 1],
                                                       dice[1, 2], dice[1, 3])))
                dice = self.fold_eval_obj.dice[np_index_vol]
                print(("after : dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                      "ED {:.2f}/{:.2f}/{:.2f}".format(dice[0, 1], dice[0, 2], dice[0, 3],
                                                       dice[1, 1], dice[1, 2], dice[1, 3])))

                hd = self.base_fold_eval_obj.hd[np_index_vol]
                print(("before: Hausdorff(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                      "ED {:.2f}/{:.2f}/{:.2f}".format(hd[0, 1], hd[1, 2],
                                                       hd[0, 3], hd[1, 1],
                                                       hd[1, 2], hd[1, 3])))
                hd = self.fold_eval_obj.hd[np_index_vol]
                print(("after : Hausdorff(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                      "ED {:.2f}/{:.2f}/{:.2f}".format(hd[0, 1], hd[1, 2], hd[0, 3],
                                                       hd[1, 1], hd[1, 2], hd[1, 3])))
            else:
                self._show_one_tissue_class_only(p_id, cardiac_phase=cardiac_phase, tissue_class=tissue_class)

    @staticmethod
    def _print_slice_performance(patient_id, ref_labels, pred_labels, classes=[1, 2, 3], comment="after"):

        for cls_idx in classes:
            cls_mean = []
            for slice_id in np.arange(ref_labels.shape[3]):
                dice = dc(pred_labels[cls_idx, :, :, slice_id], ref_labels[cls_idx, :, :, slice_id])
                cls_mean.append(dice)
                print(("INFO ({}) - Patient {}/ slice {}/cls {} {:.2f}".format(comment, patient_id, slice_id, cls_idx,
                                                                               dice)))
            print(("INFO mean={:.2f}".format(np.mean(np.array(cls_mean)))))

    def free_memory(self):
        del self.data_handler
        del self.heat_maps
        del self.corrected_pred_labels
        
    def save_corrected_labels(self, patient_id, cardiac_phase):

        if self.correct_all_seg_errors:
            output_dir = self.corr_all_pred_labels_dir
        else:
            output_dir = self.corr_pred_labels_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        if self.mc_dropout:
            file_suffix = "_{}".format(cardiac_phase) + "_mc.nii.gz"
        else:
            file_suffix = "_{}".format(cardiac_phase) + ".nii.gz"

        filename = patient_id + file_suffix
        fname = os.path.join(output_dir, filename)
        save_nifty(self.data_handler['corrected_pred_labels'].astype(np.int32), self.data_handler['voxel_spacing'],
                   fname, direction=None, origin=None)


def run_for_all_folds(root_dir, seg_model, loss_function, io_channel,
                      patient_ids=None, correct_all_seg_errors=False, do_save=True):

    src_exper_dir = os.path.expanduser(root_dir + os.sep + seg_model + "_" + loss_function)
    all_folds_settings = {'network': seg_model, 'input_channels': io_channel, }
    evaluate_baseline = False if correct_all_seg_errors else True
    save_new_labels = False if correct_all_seg_errors else True

    sim_expert = SimulateExpert(src_exper_dir, all_folds_settings=all_folds_settings,
                                patient_ids=patient_ids, verbose=False,
                                correct_all_seg_errors=correct_all_seg_errors)
    sim_expert.evaluate_detections(evaluate_baseline=evaluate_baseline, save_new_labels=save_new_labels)
    if do_save:
        sim_expert.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run manual correction of segmentation mask based on detection')
    parser.add_argument('src_path_data', type=str, help='directory that contains data objects from step 2')  # e.g. ~/expers/redo_expers/
    parser.add_argument('--seg_model', type=str, help='directory for experiment outputs')
    parser.add_argument('--loss_function', type=str, default="ce")
    parser.add_argument('--channels', type=str, choices=['bmap', 'emap', None], default=None)
    # parser.add_argument('--all_errors', action='store_true')
    args = parser.parse_args()
    do_save = True
    patient_ids = None  # ["patient016", "patient029"]
    root_dir = args.src_path_data
    # ["patient018", "patient029", "patient037", "patient057", "patient097", "patient099"]

    if args.channels is None:
        input_channels = ['emap', 'bmap']
    else:
        input_channels = [args.channels]

    for io_channel in input_channels:
        for correct_all_seg_errors in [True, False]:
            print("INFO - processing: {} - {} - {}: correct all errors: {}".format(args.seg_model, args.loss_function, io_channel,
                                                                                   correct_all_seg_errors))
            run_for_all_folds(root_dir, args.seg_model, args.loss_function, io_channel, patient_ids,
                              correct_all_seg_errors=correct_all_seg_errors, do_save=do_save)
