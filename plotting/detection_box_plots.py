from plotting.settings import plot_settings
import matplotlib.pyplot as plt
import os

import numpy as np


def make_volume_box_plots(list_of_phase_tissue_dicts, list_labels, do_show=True, do_save=False,
                          perf_type="DSC", output_dir=None, width=16, height=14, file_suffix="",
                          type_of_map=None, x_label_mask=None, model_tag=None):
    """

    :param list_of_phase_tissue_dicts:
    :param list_labels:
    :param do_show:
    :param do_save:
    :param perf_type:
    :param output_dir:
    :param width:
    :param height:
    :param file_suffix:
    :param type_of_map:
    :param x_label_mask: boolean mask of length of x-labels (so depends on model evaluation)
    :return:
    """
    # intitialize lists
    es_rv_list, es_myo_list, es_lv_list, ed_rv_list, ed_myo_list, ed_lv_list = [], [], [], [], [], []
    rv_list_labels, myo_list_labels, lv_list_labels = [], [], []
    # phase_tissue_dict is dict of dict: key1=ES/ED; key2=RV/MYO/LV
    columns = 2
    rows = 2
    width = width
    height = height

    fill_color = ['xkcd:periwinkle', 'xkcd:sea green', 'xkcd:sun yellow', 'xkcd:greeny grey', 'xkcd:brick orange']
    num_of_models = len(list_of_phase_tissue_dicts)

    for i, phase_tissue_dict in enumerate(list_of_phase_tissue_dicts):
        es_rv_list.append(phase_tissue_dict["ES"]["RV"])
        rv_list_labels.append(list_labels[i] + " RV")
        es_myo_list.append(phase_tissue_dict["ES"]["MYO"])
        myo_list_labels.append(list_labels[i] + " MYO")
        es_lv_list.append(phase_tissue_dict["ES"]["LV"])
        lv_list_labels.append(list_labels[i] + " LV")

        ed_rv_list.append(phase_tissue_dict["ED"]["RV"])
        ed_myo_list.append(phase_tissue_dict["ED"]["MYO"])
        ed_lv_list.append(phase_tissue_dict["ED"]["LV"])
    # ES-RV
    cardiac_phase = 'ES'

    if type_of_map is not None:
        mytitle = "{}-{} ({})".format(perf_type, cardiac_phase, type_of_map)
    else:
        mytitle = "{}-{}".format(perf_type, cardiac_phase)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle("{}".format(mytitle), **plot_settings.title_font_medium)
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)

    es_vector_list = es_rv_list + es_myo_list + es_lv_list
    ed_vector_list = ed_rv_list + ed_myo_list + ed_lv_list
    tissue_list_labels = rv_list_labels + myo_list_labels + lv_list_labels
    bp = ax1.boxplot(es_vector_list, labels=tissue_list_labels, showmeans=True, patch_artist=True)
    # ax1.set_title("ES", **plot_settings.title_font_medium)
    ax1.set_ylabel(perf_type, **plot_settings.axis_font20)
    ax1.set_ylim([0, 1.] if perf_type in ['surf-DSC', 'DSC', 'slice-DSC'] else [0., 60])
    plt.xticks(rotation=90)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    # set x-labels to red in case this group is significantly different than baseline, only if mask is not None
    if x_label_mask[cardiac_phase] is not None:
        index_mask = np.nonzero(x_label_mask[cardiac_phase])[0]
        for i in index_mask:
            ax1.get_xticklabels()[i].set_color("red")

    for i, flier in enumerate(bp['fliers']):
        flier.set(marker='o', color='red', alpha=0.5)
        flier.set_markerfacecolor('red')
    for i, patch in enumerate(bp['boxes']):
        j = i // num_of_models
        patch.set_alpha(0.3)
        patch.set_facecolor(fill_color[j])

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        if output_dir is None:
            raise ValueError("ERROR - parameter output_dir is None. You can't save the figure.")
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        fig_file_name = os.path.join(output_dir, file_suffix + "_" + perf_type + "_" + cardiac_phase + "_box_plot.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_file_name))

    # plt.axvline(x=0.22058956)
    # ED-RV
    cardiac_phase = 'ED'
    if type_of_map is not None:
        mytitle = "{}-{} ({})".format(perf_type, cardiac_phase, type_of_map)
    else:
        mytitle = "{}-{}".format(perf_type, cardiac_phase)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle("{}".format(mytitle), **plot_settings.title_font_medium)
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)

    bp = ax1.boxplot(ed_vector_list, labels=tissue_list_labels, showmeans=True, patch_artist=True)
    # ax1.set_title("ED", **plot_settings.title_font_medium)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylabel(perf_type, **plot_settings.axis_font20)
    plt.xticks(rotation=90)
    ax1.set_ylim([0, 1.] if perf_type in ['surf-DSC', 'DSC', 'slice-DSC'] else [0., 60])
    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.5)
        flier.set_markerfacecolor('red')
    for i, patch in enumerate(bp['boxes']):
        j = i // num_of_models
        patch.set_alpha(0.3)
        patch.set_facecolor(fill_color[j])
    # set x-labels to red in case this group is significantly different than baseline, only if mask is not None
    if x_label_mask[cardiac_phase] is not None:
        index_mask = np.nonzero(x_label_mask[cardiac_phase])[0]
        for i in index_mask:
            ax1.get_xticklabels()[i].set_color("red")

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        if output_dir is None:
            raise ValueError("ERROR - parameter output_dir is None. You can't save the figure.")
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        fig_file_name = os.path.join(output_dir, file_suffix + "_" + perf_type + "_" + cardiac_phase + "_box_plot.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_file_name))

    if do_show:
        plt.show()


def cardiac_indices_box_plot(pat_cardiac_indices, do_show=True, do_save=False, output_dir=None, cardiac_index=None,
                             tissue_class='LV', width=16, height=14, group=False, ylimits=None, file_prefix=None,
                             plot_ref=False, compute_95conf=False):
    """

    :param pat_cardiac_indices: dict: patient_id. which contains another dict with keys 'RV' and 'LV' and 'group' (DCM,ARV..)
            each sub-dict contains a numpy array of shape [#samples, 3]
            Where the three metrics are as follows:
                                                    ES volume (/BodySurfaceIndex),
                                                    ED volume (/BodySurfaceIndex),
                                                    Ejection Fraction
    :param do_show:
    :param do_save:
    :param output_dir:
    :param cardiac_index: 'EJ', 'EDV'=volume ED, 'ESV' volume ES, 'MYO_MASS' myocardial mass
    :param tissue_class: 'LV', 'RV', 'MYO'
    :param width:
    :param height:
    :param plot_ref: boolean, if true plot reference index to compare with (based on reference labels)
    :param compute_95conf: boolean, if true computes 95% conf interval for the specific metric
    :return:
    """
    assert cardiac_index in ['EF', 'ESV', 'EDV', 'MYO']
    if do_save and file_prefix is None:
        raise ValueError("file_prefix can't be None if you want to save.")
    if cardiac_index == "MYO":
        c_index = 0
    if cardiac_index == 'EF':
        c_index = 2
    elif cardiac_index == 'EDV':
        c_index = 1
        ylimits = [0, 150]
    elif cardiac_index == 'ESV':
        c_index = 0  # 'ESD'
        ylimits = [0, 150]
    columns = 4
    rows = 4
    row = 0
    fig = plt.figure(figsize=(width, height))
    fig.suptitle("{}: Cardiac index {}".format(tissue_class, cardiac_index), **plot_settings.title_font_medium)
    ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=4)
    if group:
        pat_vectors = {}
        pat_labels = []
        pat_ref_values = []
    else:
        pat_vectors = []
        pat_labels = []
        pat_ref_values = []
    max_value = 0
    metric_95conf = []
    metric_mean = []
    for patient_id, pat_data in pat_cardiac_indices.items():
        p_group = pat_data['group']
        if not isinstance(pat_data[tissue_class], np.ndarray):
            pat_data[tissue_class] = np.array(pat_data[tissue_class])
        metric = pat_data[tissue_class][..., c_index]
        metric_95conf.append(np.std(metric) * 2)  # stddev * 2 to capture 95% interval
        metric_mean.append(np.mean(metric))
        if np.max(metric) > max_value:
            max_value = np.max(metric)
        if group and p_group not in pat_vectors:
            pat_vectors[p_group] = []
            pat_labels.append(p_group)
        if group:
            pat_vectors[p_group].extend(metric)
        else:
            pat_vectors.append(metric)
            pat_labels.append(patient_id.strip('patient') + ' (' + p_group + ')')
            if "ref_"+ tissue_class in list(pat_data.keys()):
                if isinstance(pat_data["ref_"+tissue_class], np.ndarray):
                    pat_ref_values.append(pat_data["ref_"+tissue_class][c_index])
                else:
                    pat_ref_values.append(pat_data["ref_" + tissue_class])
    if group:
        pat_vectors = [a for g, a in pat_vectors.items()]
        print(len(pat_vectors))

    bp = ax1.boxplot(pat_vectors, labels=pat_labels, showmeans=False, patch_artist=True, notch=True)
    if not group and plot_ref:
        ax1.plot(np.arange(1.4, len(pat_ref_values) + 1), pat_ref_values, color='blue', marker='*', markersize=9,
                 linestyle="")
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_xlim([0, 26])
    if ylimits is None:
        ax1.set_ylim([0, max_value * 1.05])
    else:
        ax1.set_ylim(ylimits)

    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.5)
        flier.set_markerfacecolor('red')

    fill_color = ['xkcd:periwinkle', 'xkcd:sea green', 'xkcd:sun yellow', 'xkcd:greeny grey',  'xkcd:brick orange']
    if group:
        for i, patch in enumerate(bp['boxes']):
            patch.set_alpha(0.6)
            patch.set_facecolor(fill_color[i])
    else:
        for i, patch in enumerate(bp['boxes']):
            patch.set_alpha(0.6)
            patch.set_facecolor(fill_color[int(i / 5)])

    if do_save:
        if output_dir is None:
            raise ValueError("ERROR - parameter output_dir is None. You can't save the figure.")
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        file_prefix += "_" + cardiac_index + "_" + tissue_class
        fig_file_name = os.path.join(output_dir, file_prefix + "_boxplot.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_file_name))
    if do_show:
        plt.show()
    plt.close()

    if compute_95conf:
        return np.array(metric_95conf), np.array(metric_mean)


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


def plot_volume_hist(pat_frame_ids, list_result_objects, width=16, height=12, perf_type=None, res_label=None,
                     file_prefix=None, output_dir=None, do_save=False, bg_classes=(0, 4), patient_ids=None,
                     type_of_map=None):
    """
        list_result_objects: list of numpy file archive objects loaded with SimulateExpert.load_results(as_test_result=False)

        IMPORTANT: For the p-value calculation we ASSUME that list_result_objects contains 3 objects and the
        order should be (1) segmentation baseline (2) seg-results when all "low-quality" segs would be corrected
                        (3) seg-results when all errors in regions detected would be corrected

    """

    pat_ids = np.unique(pat_frame_ids[:, 0].astype(np.int))
    rows, columns = 4, 2
    fig = plt.figure(figsize=(width, height))
    if type_of_map is not None:
        mytitle = perf_type + " ({})".format(type_of_map)
    else:
        mytitle = perf_type
    fig.suptitle("{} for patient volumes".format(mytitle), **plot_settings.title_font_medium)
    ax2 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax1 = plt.subplot2grid((rows, columns), (2, 0), rowspan=2, colspan=2)
    plot_colors = [None, 'xkcd:emerald green', 'xkcd:crimson', 'xkcd:royal blue']
    tissue_classes = ['', 'RV', 'Myo', 'LV']
    diff_x = 0
    x_values = []
    x_labels = []
    plot_gap = 1 / len(list_result_objects)
    for idx, res_obj in enumerate(list_result_objects):
        # slice_stats_es. slice_stats_ed has shape [all_slices, nclasses] e.g. for slice-HD or slice-DSC
        list_vectors, vol_stats_es, _ = split_by_tissue_class(res_obj, perf_type)
        num_of_pats, nclasses = vol_stats_es.shape
        for cls_idx in range(nclasses):
            if cls_idx not in bg_classes:
                x = cls_idx + diff_x
                x_values.append(x)
                x_labels.append(res_label[idx] + "-" + tissue_classes[cls_idx])
                x_vals = np.tile(x, num_of_pats)
                ax1.scatter(x_vals, list_vectors["ES"][cls_idx],  alpha=0.5, color=plot_colors[cls_idx])
                ax2.scatter(x_vals, list_vectors["ED"][cls_idx], alpha=0.5, color=plot_colors[cls_idx])
                for i in range(num_of_pats):
                    if patient_ids is None or pat_ids[i] in patient_ids:
                        ax1.text(x_vals[i], list_vectors["ES"][cls_idx][i], pat_ids[i])
                        ax2.text(x_vals[i], list_vectors["ED"][cls_idx][i], pat_ids[i])
        # next result object is plotted slightly next to previous
        diff_x += plot_gap
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_labels, rotation=90)
    # ax1.legend(loc=0, prop={'size': 20})
    ax1.set_ylabel(perf_type, **plot_settings.axis_font20)
    ax1.set_xlabel("Tissue classes", **plot_settings.axis_font20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.set_title("ED", **plot_settings.title_font_medium)
    ax2.set_xticks(x_values)
    ax2.set_xticklabels(x_labels, rotation=90)
    # ax2.legend(loc=0, prop={'size': 20})
    ax2.set_ylabel(perf_type, **plot_settings.axis_font20)
    ax2.set_xlabel("Tissue classes", **plot_settings.axis_font20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    title_suffix_ed, title_suffix_es = "", ""
    ax1.set_title("ES" + title_suffix_es, **plot_settings.title_font_medium)
    ax2.set_title("ED" + title_suffix_ed, **plot_settings.title_font_medium)
    # plt.text(0.02, 0.5, es_p_values, fontsize=14, transform=plt.gcf().transFigure)

    fig.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])
    if do_save:
        if output_dir is None:
            raise ValueError("ERROR - parameter output_dir is None. You can't save the figure.")
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        fig_file_name = os.path.join(output_dir, file_prefix + "_{}_pat_volumes.pdf".format(perf_type))

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_file_name))

    plt.show()
