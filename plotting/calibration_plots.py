
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
from utils.common import translate_combined_model_tag


def plot_reliability_diagram(cal_data, height=None, width=16, do_show=True, do_save=False,
                             per_class=False):

    """

    :param cal_data:
           'prob_bins'
           'acc_bins'
           'mean_ece_per_class'
            'probs_per_bin'
            'load_dir'
            'cardiac_phase'
            'mc_dropout'
            'with_bg'
            'nclasses'

    :param do_show:
    :param do_save:
    :param per_class: plot per class of one figure over mean class values
    :return:
    """
    cardiac_phase, mc_dropout, nclasses = cal_data['cardiac_phase'], cal_data['mc_dropout'], cal_data['nclasses']
    cls_labels = ["BG", "RV", "MYO", "LV"]
    if per_class:
        rows = 2 * 3
        columns = 4
        if height is None:
            height = 14
        the_range = np.arange(1, nclasses)
        acc_bins = cal_data['acc_per_bin']
    else:
        # 2 figures (ES/ED) average values over classes
        rows = 2
        columns = 2
        if height is None:
            height = 10
        the_range = np.arange(0, 1)
        # average per phase over classes ignoring background class

        acc_bins = np.mean(cal_data['acc_per_bin'][1:], axis=0)
        probs_per_bin = np.mean(cal_data['probs_per_bin'][1:], axis=0)

    row = 0
    bar_width = 0.09
    tick_size = 24
    legend_size = 24
    axis_label_size = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}
    sub_title_size = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}
    prob_bins = cal_data['prob_bin_edges']
    # assuming output dir has syntax: /home/jorg/expers/dcnn_mc_< bla bla>. We use the path to extract the model name
    output_dir = cal_data['load_dir']
    model_specs = (output_dir.split(os.sep)[-1]).split("_")
    network = translate_combined_model_tag(model_specs[0] + "_" + model_specs[1] + "_" + model_specs[2])
    dropout = "_mc" if cal_data['mc_dropout'] else ""
    model_info = "{} ({})".format(network, "MC dropout") if cal_data['mc_dropout'] else \
        "{}".format(network)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(model_info, **{'fontname': 'Monospace', 'size': '30', 'color': 'black', 'weight': 'normal'})

    for cls_idx in the_range:
        ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
        if per_class:
            acc_ed = acc_bins[cls_idx]
            # ax1.set_title("{} - class {}".format(cardiac_phase, cls_labels[cls_idx]), **sub_title_size)
        else:
            acc_ed = acc_bins
            # print(acc_ed)
            # ax1.set_title("{} ".format(cardiac_phase), **sub_title_size)
        # compute the gap between fraction of forecast and identity.
        # old version how we computed miscalibration (wrong): y_gaps = np.abs(prob_bins[1:] - acc_ed)
        # The calibration gap is the absolute difference between the accuracy for a particular bin MINUS the mean
        # confidence in that bin i.e. the mean softmax probability (verified this 4/07
        y_gaps = np.abs(probs_per_bin - acc_ed)

        ax1.bar(prob_bins[1:], acc_ed, bar_width, color="b", alpha=0.4)
        # ax1.bar(prob_bins[1:], y_gaps, bar_width, bottom=acc_ed, color="r", alpha=0.3, hatch='/',
        #        label="Miscalibration", edgecolor='red')
        ax1.bar(prob_bins[1:], y_gaps, bar_width, color="r", alpha=0.3, hatch='/',
                label="Miscalibration", edgecolor='red')
        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1.1])
        ax1.set_ylabel("Fraction of positive cases", **axis_label_size)
        ax1.set_xlabel("Probability", **axis_label_size)
        ax1.set_xticks(np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        ax1.tick_params(axis='both', which='major', labelsize=tick_size)
        # plot the identify function i.e. bisector line. The mean identity means that the accuracies should
        # be equal to the mean probabilities in the bin.
        # ax1.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="gray", linewidth=6)
        ax1.plot(prob_bins, prob_bins, "--", c="gray", linewidth=6)
        ax1.legend(loc=0, prop={'size': legend_size})

        row += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        fig_name = "prob_calibration_" + network + "_" + cal_data['cardiac_phase'] + dropout

        fig_name = os.path.join(output_dir, fig_name + ".jpeg")
        plt.savefig(fig_name, bbox_inches='tight')
        plt.savefig(fig_name.replace(".jpeg", ".pdf"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_name))
    if do_show:
        plt.show()

    plt.close()
