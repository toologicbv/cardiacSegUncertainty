import numpy as np
import os
from utils.common import NETWORK_WITH_LOSS_LABEL
from eval_detection.helpers import compute_p_values


def load_results(src_path, mc_dropout=False, num_fold=4, file_prefix="results", num_patients=25):
    """

    :param src_path: e.g. "~/expers/acdc/*/drn_mc_ce"
    :param mc_dropout:
    :return:
    """

    model = src_path.split(os.sep)[-1]
    model = NETWORK_WITH_LOSS_LABEL[model]
    if mc_dropout:
        model += "+MC"
    results = {"mean_dice_es": np.zeros(4), "mean_dice_ed": np.zeros(4), "std_dice_es": np.zeros(4), "std_dice_ed": np.zeros(4),
               "mean_hd_es": np.zeros(4), "mean_hd_ed": np.zeros(4), "std_hd_es": np.zeros(4),
               "std_hd_ed": np.zeros(4), 'dice_es': np.empty((0, 4)), 'dice_ed': np.empty((0, 4)),
               'hd_es': np.empty((0, 4)), 'hd_ed': np.empty((0, 4))}
    # NOTE: we define the dictionaries that need special processing: concatenate results along dimension 0
    #       and do not average
    archive_to_concat = ['dice_es', 'dice_ed', 'hd_es', 'hd_ed']
    for f in np.arange(num_fold):

        if mc_dropout:
            filename = file_prefix + "_f{}_{}_mc.npz".format(f, num_patients)
        else:
            filename = file_prefix + "_f{}_{}.npz".format(f, num_patients)
        load_path = os.path.join(os.path.expanduser(src_path.replace("*", "f" + str(f))), filename)
        print("INFO - {}".format(load_path))

        loaded_arrays = np.load(load_path, allow_pickle=True)

        for archive in loaded_arrays.files:
            if archive in results.keys() and archive not in archive_to_concat:
                results[archive] += loaded_arrays[archive]
            if archive in archive_to_concat:
                results[archive] = np.concatenate((results[archive], loaded_arrays[archive])) if results[archive].ndim else loaded_arrays[archive]

    for key in results.keys():
        # do not average the arrays that contain individual measurements: dice_es, hd_es...
        if key not in archive_to_concat:
            results[key] = results[key] * 1/4

    msg_dsc, msg_hd = format_result(results, model)

    return results, msg_dsc, msg_hd


def load_results_sim_correction(src_path, mc_dropout=False, result_type='detection', base_results=None):
    """

    :param src_path: we assume src_path is something like "~/expers/acdc/drn_mc_dice/
    :param result_type: base = segmentation-only
                        det_base = baseline for combined segmentation & detection
                        detection = results combined approach
    :param mc_dropout:
    :return:
    """
    assert result_type in ['det_base', 'detection']
    model = src_path.split(os.sep)[-1]
    model = NETWORK_WITH_LOSS_LABEL[model]

    f_suffix = "_mc.npz" if mc_dropout else ".npz"
    if result_type == 'det_base':
        filename = "results_fall_corrall_100" + f_suffix
    else:
        filename = "results_fall_100" + f_suffix
    load_path = os.path.join(os.path.expanduser(src_path), filename)
    loaded_arrays = np.load(load_path, allow_pickle=True)
    perf_types = ['HD', 'DSC']
    p_values = {'DSC': None, 'HD': None}
    if base_results is not None:
        for perf_type in perf_types:
            p_values[perf_type] = compute_p_values(base_results, loaded_arrays, perf_type)

    msg_dsc, msg_hd = format_result(loaded_arrays, model, p_values=p_values, p_threshold=0.05,
                                    result_type=result_type)
    print("Load results from {}".format(load_path))
    return loaded_arrays, msg_dsc, msg_hd


def format_result(np_result_obj, model_name=None, p_values=None, p_threshold=0.05, result_type=None):
    """

    :param np_result_obj: Numpy file archive that holds all numpy arrays as defined in TestResult object (see below)
    :param model_name
    :param p_values: dictionary with keys: 'HD' and 'DSC'
                     contains another dictionary with keys: 'ED' and 'ES'.
                     and each value contains numpy array of size [4]
    :param p_values_hd: same as above
    :param p_threshold
    Important: Note that we assume that all number array containing the nclass dimension (tissue structures)
               that indices: 0=background 1=RV, 2=Myo/LVM, 3=LV

    :return:
    """
    def prepare_p_value_str(p_values, cardiac_phase):
        # Note: we mark DSC or HD results with a "*" in case they are statistically significant
        if p_values is None:
            p_value_str_dsc, p_value_str_hd = [""] * 4, [""] * 4
        else:
            p_values_dsc, p_values_hd = p_values['DSC'], p_values['HD']
            p_value_str_dsc, p_value_str_hd = [], []
            for p_dsc, p_hd in zip(p_values_dsc[cardiac_phase], p_values_hd[cardiac_phase]):
                p_value_str_dsc.append("*" if p_dsc < p_threshold else "")
                p_value_str_hd.append("*" if p_hd < p_threshold else "")
        return p_value_str_dsc, p_value_str_hd

    # our latex table contains two columns that specify whether we show the "segmentation detection baseline" or
    # actual performance of "combined segmentation & detection" approach
    det_base = ""
    detection = ""
    if result_type is not None:
        model_name = ""
        if result_type == "det_base":
            det_base = "\\textbf{x}"
        else:
            detection = "\\textbf{x}"
    else:
        # segmentation only
        pass

    key = 'mean_dice_ed'
    dice_mean, hd_mean = np_result_obj[key], np_result_obj[key.replace("dice", "hd")]
    key_std = key.replace("mean", "std")
    dice_std, hd_std = np_result_obj[key_std], np_result_obj[key_std.replace("dice", "hd")]
    p_value_str_dsc, p_value_str_hd = prepare_p_value_str(p_values, cardiac_phase="ED")

    msg_dsc = " & {} & {} & {} & {}{:.3f}$\pm${:.2f} & {}{:.3f}$\pm${:.2f} & {}{:.3f}$\pm${:.2f} " \
              "".format(model_name, det_base, detection, p_value_str_dsc[3], dice_mean[3], dice_std[3],
                    p_value_str_dsc[1], dice_mean[1], dice_std[1],
                    p_value_str_dsc[2], dice_mean[2], dice_std[2])

    msg_hd = " & {} & {} & {} &  {}{:.1f}$\pm${:.1f} & {}{:.1f}$\pm${:.1f} & {}{:.1f}$\pm${:.1f}" \
             "".format(model_name, det_base, detection, p_value_str_hd[3], hd_mean[3], hd_std[3], p_value_str_hd[1],
                       hd_mean[1], hd_std[1], p_value_str_hd[2], hd_mean[2], hd_std[2])

    key = 'mean_dice_es'
    dice_mean, hd_mean = np_result_obj[key], np_result_obj[key.replace("dice", "hd")]
    key_std = key.replace("mean", "std")

    dice_std, hd_std = np_result_obj[key_std], np_result_obj[key_std.replace("dice", "hd")]
    p_value_str_dsc, p_value_str_hd = prepare_p_value_str(p_values, cardiac_phase="ES")

    msg_dsc += " & {}{:.3f}$\pm${:.2f} & {}{:.3f}$\pm${:.2f} & {}{:.3f}$\pm${:.2f} \\\\ " \
              "".format(p_value_str_dsc[3], dice_mean[3], dice_std[3],
                        p_value_str_dsc[1], dice_mean[1], dice_std[1],
                        p_value_str_dsc[2], dice_mean[2], dice_std[2])

    msg_hd += " & {}{:.1f}$\pm${:.1f} & {}{:.1f}$\pm${:.1f} & {}{:.1f}$\pm${:.1f} \\\\" \
             "".format(p_value_str_hd[3], hd_mean[3], hd_std[3], p_value_str_hd[1], hd_mean[1], hd_std[1],
                       p_value_str_hd[2], hd_mean[2], hd_std[2])
    return msg_dsc, msg_hd


class TestResult(object):

    def __init__(self):
        self.dice = {}
        self.slice_dice = {}
        self.surf_dice = {}
        self.hd = {}
        self.hd95 = {}
        self.assd = {}
        self.slice_hd = {}
        self.cardiac_phases = {}
        self.stats_dice = None
        self.stats_slice_dice = None
        self.stats_surf_dice = None
        self.stats_hd = None
        self.stats_hd95 = None
        self.stats_assd = None
        self.stats_slice_hd = None
        self.pat_ids = []
        # We use this property only when we load the TestResult object after evaluation of the
        # segmentation results using a human-in-the-loop setting
        self.pat_frame_ids = None
        self.network = None
        self.loss_function = None
        self.fold = None

    def __call__(self, dice, cardiac_phase_tag, hd=None, surf_dice=None, slice_dice=None, slice_hd=None,
                 pat_id=None, hd95=None, assd=None):
        if pat_id is not None:
            if pat_id not in self.pat_ids:
                self.pat_ids.append(pat_id)

        if cardiac_phase_tag not in self.cardiac_phases.keys():
            # print("Test_result: new phase {}".format(cardiac_phase_tag))
            self.cardiac_phases[cardiac_phase_tag] = cardiac_phase_tag
            self.dice[cardiac_phase_tag] = np.expand_dims(dice, axis=0)
            # Please note: for the slice_dice and slice_hd we don't extend the np array's dimension
            # because both have shape [#slices, #classes] and we just concatenate along the first dimension
            if hd is not None:
                self.hd[cardiac_phase_tag] = np.expand_dims(hd, axis=0)
            if hd95 is not None:
                self.hd95[cardiac_phase_tag] = np.expand_dims(hd95, axis=0)
            if assd is not None:
                self.assd[cardiac_phase_tag] = np.expand_dims(assd, axis=0)
            if surf_dice is not None:
                self.surf_dice[cardiac_phase_tag] = np.expand_dims(surf_dice, axis=0)
            if slice_dice is not None:
                self.slice_dice[cardiac_phase_tag] = slice_dice
            if slice_hd is not None:
                self.slice_hd[cardiac_phase_tag] = slice_hd
        else:
            self.dice[cardiac_phase_tag] = np.vstack([self.dice[cardiac_phase_tag], dice])
            if hd is not None:
                self.hd[cardiac_phase_tag] = np.vstack([self.hd[cardiac_phase_tag], hd])
            if hd95 is not None:
                self.hd95[cardiac_phase_tag] = np.vstack([self.hd95[cardiac_phase_tag], hd95])
            if assd is not None:
                self.assd[cardiac_phase_tag] = np.vstack([self.assd[cardiac_phase_tag], assd])
            if surf_dice is not None:
                self.surf_dice[cardiac_phase_tag] = np.vstack([self.surf_dice[cardiac_phase_tag], surf_dice])
            if slice_dice is not None:
                self.slice_dice[cardiac_phase_tag] = np.vstack([self.slice_dice[cardiac_phase_tag], slice_dice])
            if slice_hd is not None:
                self.slice_hd[cardiac_phase_tag] = np.vstack([self.slice_hd[cardiac_phase_tag], slice_hd])

    def get_dices(self):
        if len(self.dice) == 2:
            np_array = np.concatenate((self.dice['ES'], self.dice['ED']))
            return np_array
        else:
            return next(iter(self.dice.values()))

    def get_slice_dices(self):
        if len(self.slice_dice) == 2:
            np_array = np.concatenate((self.slice_dice['ES'], self.slice_dice['ED']))
            return np_array
        else:
            return next(iter(self.slice_dice.values()))

    def get_hds(self):
        if len(self.hd) == 2:
            np_array = np.concatenate((self.hd['ES'], self.hd['ED']))
            return np_array
        else:
            return next(iter(self.hd.values()))

    def get_hds95(self):
        if len(self.hd95) == 2:
            np_array = np.concatenate((self.hd95['ES'], self.hd95['ED']))
            return np_array
        else:
            return next(iter(self.hd95.values()))

    def get_assd(self):
        if len(self.assd) == 2:
            np_array = np.concatenate((self.assd['ES'], self.assd['ED']))
            return np_array
        else:
            return next(iter(self.assd.values()))

    def get_slice_hds(self):
        if len(self.slice_hd) == 2:
            np_array = np.concatenate((self.slice_hd['ES'], self.slice_hd['ED']))
            return np_array
        else:
            return next(iter(self.slice_hd.values()))

    def get_surf_dices(self):
        if len(self.surf_dice) == 2:
            np_array = np.concatenate((self.surf_dice['ES'], self.surf_dice['ED']))
            return np_array
        else:
            return next(iter(self.surf_dice.values()))

    def get_stats_dice(self):
        if self.stats_dice is None:
            self.compute_stats()
        if len(self.stats_dice) == 2:
            np_mean = np.concatenate((self.stats_dice['ES']['mean'], self.stats_dice['ED']['mean']))
            np_std = np.concatenate((self.stats_dice['ES']['std'], self.stats_dice['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_dice.values()))

    def get_stats_slice_dice(self):
        if self.stats_slice_dice is None:
            self.compute_stats()
        if len(self.stats_slice_dice) == 2:
            np_mean = np.concatenate((self.stats_slice_dice['ES']['mean'], self.stats_slice_dice['ED']['mean']))
            np_std = np.concatenate((self.stats_slice_dice['ES']['std'], self.stats_slice_dice['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_slice_dice.values()))

    def get_stats_surf_dice(self):
        if self.stats_surf_dice is None:
            self.compute_stats()
        if len(self.stats_surf_dice) == 2:
            np_mean = np.concatenate((self.stats_surf_dice['ES']['mean'], self.stats_surf_dice['ED']['mean']))
            np_std = np.concatenate((self.stats_surf_dice['ES']['std'], self.stats_surf_dice['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_surf_dice.values()))

    def get_stats_hd(self):
        if self.stats_hd is None:
            self.compute_stats()
        if len(self.stats_hd) == 2:
            np_mean = np.concatenate((self.stats_hd['ES']['mean'], self.stats_hd['ED']['mean']))
            np_std = np.concatenate((self.stats_hd['ES']['std'], self.stats_hd['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_hd.values()))

    def get_stats_hd95(self):
        if self.stats_hd95 is None:
            self.compute_stats()
        if len(self.stats_hd95) == 2:
            np_mean = np.concatenate((self.stats_hd95['ES']['mean'], self.stats_hd95['ED']['mean']))
            np_std = np.concatenate((self.stats_hd95['ES']['std'], self.stats_hd95['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_hd.values()))

    def get_stats_assd(self):
        if self.stats_assd is None:
            self.compute_stats()
        if len(self.stats_assd) == 2:
            np_mean = np.concatenate((self.stats_assd['ES']['mean'], self.stats_assd['ED']['mean']))
            np_std = np.concatenate((self.stats_assd['ES']['std'], self.stats_assd['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_hd.values()))

    def get_stats_slice_hd(self):
        if self.stats_slice_hd is None:
            self.compute_stats()
        if len(self.stats_slice_hd) == 2:
            np_mean = np.concatenate((self.stats_slice_hd['ES']['mean'], self.stats_slice_hd['ED']['mean']))
            np_std = np.concatenate((self.stats_slice_hd['ES']['std'], self.stats_slice_hd['ED']['std']))
            return {'mean': np_mean, 'std': np_std}
        else:
            return next(iter(self.stats_slice_hd.values()))

    def compute_stats(self):
        self.stats_dice = {}
        self.stats_hd = {}
        self.stats_hd95 = {}
        self.stats_assd = {}
        self.stats_surf_dice = {}
        self.stats_slice_dice = {}
        self.stats_slice_hd = {}
        for phase_id, dice in self.dice.items():
            self.stats_dice[phase_id] = {'mean': np.mean(dice, axis=0), 'std': np.std(dice, axis=0)}
            if len(self.hd) != 0:
                hd = self.hd[phase_id]
                self.stats_hd[phase_id] = {'mean': np.mean(hd, axis=0), 'std': np.std(hd, axis=0)}
            if len(self.hd95) != 0:
                hd95 = self.hd95[phase_id]
                self.stats_hd95[phase_id] = {'mean': np.mean(hd95, axis=0), 'std': np.std(hd95, axis=0)}
            if len(self.assd) != 0:
                assd = self.assd[phase_id]
                self.stats_assd[phase_id] = {'mean': np.mean(assd, axis=0), 'std': np.std(assd, axis=0)}
            if len(self.surf_dice) != 0:
                surf_dice = self.surf_dice[phase_id]
                self.stats_surf_dice[phase_id] = {'mean': np.mean(surf_dice, axis=0), 'std': np.std(surf_dice, axis=0)}
            if len(self.slice_dice) != 0:
                slice_dice = self.slice_dice[phase_id]
                self.stats_slice_dice[phase_id] = {'mean': np.mean(slice_dice, axis=0), 'std': np.std(slice_dice, axis=0)}

    @property
    def phases(self):
        # returns list of cardiac phases
        if len(self.cardiac_phases) == 2:
            return "".join([ph for ph in self.cardiac_phases.keys()])
        else:
            return list(self.cardiac_phases)[0]

    def generate_box_plot_input(self):
        raise NotImplementedError

    def get_boxplot_vectors(self):
        self.generate_box_plot_input()
        return self.bp_list_dice, self.bp_list_hd


class ACDCTestResult(TestResult):

    def __init__(self):
        super().__init__()

    def generate_box_plot_input(self):
        if self.phases == "ESED" and len(self.cardiac_phases) == 1:
            dice_dict, hd_dict = self._split_combined_phases()
        else:
            dice_dict, hd_dict, surf_dice_dict, slice_dice_dict, slice_hd_dict = self.dice, self.hd, self.surf_dice, \
                                                                                 self.slice_dice, self.slice_hd
        self.bp_list_dice = {}
        self.bp_list_slice_dice = {}
        self.bp_list_hd = {}
        self.bp_list_slice_hd = {}
        self.bp_list_surf_dice = {}
        for phase, dice in dice_dict.items():
            self.bp_list_dice[phase] = {'RV': None, 'MYO': None, 'LV': None}
            # self.dice[phase] has shape [#patient, 4] 0=BG, 1=RV, 2=MYO, 3=LV
            self.bp_list_dice[phase]["RV"] = dice[:, 1]
            self.bp_list_dice[phase]["MYO"] = dice[:, 2]
            self.bp_list_dice[phase]["LV"] = dice[:, 3]

        for phase, hd in hd_dict.items():
            self.bp_list_hd[phase] = {'RV': None, 'MYO': None, 'LV': None}
            # same for HD
            self.bp_list_hd[phase]["RV"] = hd[:, 1]
            self.bp_list_hd[phase]["MYO"] = hd[:, 2]
            self.bp_list_hd[phase]["LV"] = hd[:, 3]

        for phase, surf_dice in surf_dice_dict.items():
            if surf_dice is not None:
                self.bp_list_surf_dice[phase] = {'RV': None, 'MYO': None, 'LV': None}
                # same for HD
                self.bp_list_surf_dice[phase]["RV"] = surf_dice[:, 1]
                self.bp_list_surf_dice[phase]["MYO"] = surf_dice[:, 2]
                self.bp_list_surf_dice[phase]["LV"] = surf_dice[:, 3]

        for phase, slice_dice in slice_dice_dict.items():
            if slice_dice is not None:
                self.bp_list_slice_dice[phase] = {'RV': None, 'MYO': None, 'LV': None}
                # same for HD
                self.bp_list_slice_dice[phase]["RV"] = slice_dice[:, 1]
                self.bp_list_slice_dice[phase]["MYO"] = slice_dice[:, 2]
                self.bp_list_slice_dice[phase]["LV"] = slice_dice[:, 3]

        for phase, slice_hd in slice_hd_dict.items():
            if slice_hd is not None:
                self.bp_list_slice_hd[phase] = {'RV': None, 'MYO': None, 'LV': None}
                # same for HD
                self.bp_list_slice_hd[phase]["RV"] = slice_hd[:, 1]
                self.bp_list_slice_hd[phase]["MYO"] = slice_hd[:, 2]
                self.bp_list_slice_hd[phase]["LV"] = slice_hd[:, 3]

    def _split_combined_phases(self):
        dice = next(iter(self.dice.values()))
        dice = {"ES": dice[:, 4:], "ED": dice[:, 4:]}
        hd = next(iter(self.hd.values()))
        hd = {"ES": hd[:, 4:], "ED": hd[:, 4:]}
        return dice, hd

    def save(self, filename):
        np.savez(filename, dice_es=self.dice["ES"], dice_ed=self.dice["ED"], hd_es=self.hd["ES"], hd_ed=self.hd["ED"],
                 mean_dice_es=self.stats_dice["ES"]['mean'], mean_dice_ed=self.stats_dice["ED"]['mean'],
                 mean_hd_es=self.stats_hd["ES"]['mean'], mean_hd_ed=self.stats_hd["ED"]['mean'],
                 std_dice_es=self.stats_dice["ES"]['std'], std_dice_ed=self.stats_dice["ED"]['std'],
                 std_hd_es=self.stats_hd["ES"]['std'], std_hd_ed=self.stats_hd["ED"]['std'],
                 hd95_es= self.hd95["ES"], hd95_ed=self.hd95["ED"],
                 mean_hd95_es=self.stats_hd95["ES"]['mean'], mean_hd95_ed=self.stats_hd95["ED"]['mean'],
                 std_hd95_es=self.stats_hd95["ES"]['std'], std_hd95_ed=self.stats_hd95["ED"]['std'],
                 mean_assd_es=self.stats_assd["ES"]['mean'], mean_assd_ed=self.stats_assd["ED"]['mean'],
                 std_assd_es=self.stats_assd["ES"]['std'], std_assd_ed=self.stats_assd["ED"]['std'],
                 ids=np.array(self.pat_ids))

    def show_results(self):
        if self.stats_dice is None:
            self.get_stats_dice()
        if self.stats_hd is None:
            self.get_stats_hd()
        for phase_id in self.stats_dice.keys():
            dice_mean = self.stats_dice[phase_id]['mean']
            print("dice(RV/Myo/LV):\t{} {:.2f}/{:.2f}/{:.2f}\t"
               "".format(phase_id, dice_mean[1], dice_mean[2], dice_mean[3]))
            if self.hd is not None:
                hd_mean = self.stats_hd[phase_id]['mean']
                print("\t\t\t\t\t"
                      "Hausdorff(RV/Myo/LV):\t{} {:.2f}/{:.2f}/{:.2f}\t"
                      "".format(phase_id, hd_mean[1], hd_mean[2],
                                hd_mean[3]))
            if self.hd95 is not None:
                hd95_mean = self.stats_hd95[phase_id]['mean']
                print("\t\t\t\t\t"
                      "Hausdorff95(RV/Myo/LV):\t{} {:.2f}/{:.2f}/{:.2f}\t"
                      "".format(phase_id, hd95_mean[1], hd95_mean[2],
                                hd95_mean[3]))
            if self.assd is not None and phase_id in self.stats_assd.keys():
                assd_mean = self.stats_assd[phase_id]['mean']
                print("\t\t\t\t\t"
                      "Avg Sym SD(RV/Myo/LV):\t{} {:.2f}/{:.2f}/{:.2f}\t"
                      "".format(phase_id, assd_mean[1], assd_mean[2],
                                assd_mean[3]))

    def excel_string(self):
        for phase_id in self.stats_dice.keys():
            dice_mean = self.stats_dice[phase_id]['mean']
            hd_mean = self.stats_hd[phase_id]['mean']

            msg = "LV/RV/Myo (DSC) \t{} {:.3f} \t {:.3f} \t {:.3f} ".format(phase_id, dice_mean[3],
                                                                                          dice_mean[1], dice_mean[2])
            print(msg)
            msg = "LV/RV/Myo (HD) \t{} {:.1f} \t {:.1f} \t {:.1f} ".format(phase_id, hd_mean[3],
                                                                                                          hd_mean[1],
                                                                                                          hd_mean[2])
            print(msg)


class ARVCTestResult(ACDCTestResult):

    def __init__(self):
        super().__init__()

    def __call__(self, dice, cardiac_phase_tag, hd=None, surf_dice=None, slice_dice=None, slice_hd=None,
                 pat_id=None, ignore_labels=None):

        if pat_id is not None:
            if pat_id not in self.pat_ids:
                self.pat_ids.append(pat_id)

        if cardiac_phase_tag not in self.cardiac_phases.keys():
            # print("Test_result: new phase {}".format(cardiac_phase_tag))
            self.cardiac_phases[cardiac_phase_tag] = cardiac_phase_tag
            self.dice[cardiac_phase_tag] = np.expand_dims(dice, axis=0)
            # Please note: for the slice_dice and slice_hd we don't extend the np array's dimension
            # because both have shape [#slices, #classes] and we just concatenate along the first dimension
            if hd is not None:
                self.hd[cardiac_phase_tag] = np.expand_dims(hd, axis=0)
            if surf_dice is not None:
                self.surf_dice[cardiac_phase_tag] = np.expand_dims(surf_dice, axis=0)
            if slice_dice is not None:
                self.slice_dice[cardiac_phase_tag] = slice_dice
            if slice_hd is not None:
                self.slice_hd[cardiac_phase_tag] = slice_hd
        else:
            self.dice[cardiac_phase_tag] = np.vstack([self.dice[cardiac_phase_tag], dice])
            if hd is not None:
                self.hd[cardiac_phase_tag] = np.vstack([self.hd[cardiac_phase_tag], hd])
            if surf_dice is not None:
                self.surf_dice[cardiac_phase_tag] = np.vstack([self.surf_dice[cardiac_phase_tag], surf_dice])
            if slice_dice is not None:
                self.slice_dice[cardiac_phase_tag] = np.vstack([self.slice_dice[cardiac_phase_tag], slice_dice])
            if slice_hd is not None:
                self.slice_hd[cardiac_phase_tag] = np.vstack([self.slice_hd[cardiac_phase_tag], slice_hd])

    def show_results(self, transfer_learning=False):
        if self.stats_dice is None:
            self.get_stats_dice()
        if self.stats_hd is None:
            self.get_stats_hd()
        if transfer_learning:
            lv_idx = 3
            rv_idx = 1
        else:
            lv_idx = 1
            rv_idx = 2
        for phase_id in self.stats_dice.keys():
            dice_mean = self.stats_dice[phase_id]['mean']
            print("dice(RV/LV):\t{} {:.2f}/{:.2f}\t"
               "".format(phase_id, dice_mean[rv_idx], dice_mean[lv_idx]))
            if self.hd is not None:
                hd_mean = self.stats_hd[phase_id]['mean']
                print("\t\t\t\t\t"
                      "Hausdorff(RV/LV):\t{} {:.2f}/{:.2f}\t"
                      "".format(phase_id, hd_mean[rv_idx], hd_mean[lv_idx]))

    def save(self, filename):
        np.savez(filename, dice_es=self.dice["ES"], dice_ed=self.dice["ED"], hd_es=self.hd["ES"], hd_ed=self.hd["ED"],
                 mean_dice_es=self.stats_dice["ES"]['mean'], mean_dice_ed=self.stats_dice["ED"]['mean'],
                 mean_hd_es=self.stats_hd["ES"]['mean'], mean_hd_ed=self.stats_hd["ED"]['mean'],
                 std_dice_es=self.stats_dice["ES"]['std'], std_dice_ed=self.stats_dice["ED"]['std'],
                 std_hd_es=self.stats_hd["ES"]['std'], std_hd_ed=self.stats_hd["ED"]['std'],
                 # hd95_es= self.hd95["ES"], hd95_ed=self.hd95["ED"],
                 # mean_hd95_es=self.stats_hd95["ES"]['mean'], mean_hd95_ed=self.stats_hd95["ED"]['mean'],
                 # std_hd95_es=self.stats_hd95["ES"]['std'], std_hd95_ed=self.stats_hd95["ED"]['std'],
                 # mean_assd_es=self.stats_assd["ES"]['mean'], mean_assd_ed=self.stats_assd["ED"]['mean'],
                 # std_assd_es=self.stats_assd["ES"]['std'], std_assd_ed=self.stats_assd["ED"]['std'],
                 ids=np.array(self.pat_ids))