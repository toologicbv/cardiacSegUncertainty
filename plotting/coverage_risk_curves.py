import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pylab
import numpy as np
import os
from utils.common import determine_plot_style
from utils.common import translate_combined_model_tag

# PLOT_COLORS = ['xkcd:emerald green', 'xkcd:bright pink', 'xkcd:royal blue', 'xkcd:dark olive',
#               'xkcd:tangerine', 'xkcd:yellowish green']

PLOT_COLORS = ['xkcd:black',  'xkcd:blue', 'xkcd:bluish green', 'xkcd:sky blue',
               'xkcd:vermillion', 'xkcd:reddish purple']

ALPHA = 0.55
LW = 3
LINESTYLES = {'drn_mc': "--", 'dcnn_mc': "-.", 'unet_mc': ":",
              'drn_mc_ce': "--", 'dcnn_mc_brier': "-.", 'unet_mc_ce': ":",
              'drn_mc_dice': "--", 'dcnn_mc_dice': "-.", 'unet_mc_dice': ":"}


def plot_coverage_risk_curve(list_eval_obj, width=10, height=8, do_save=False, do_show=True,
                             error_metric='dice', output_dir=None, eval_id=None):
    """

    :param list_eval_obj: list of numpy archive objects with the following dict entries:
                            'coverage' [#thresholds],
                            'dice' [#patients,  #thresholds, nclasses]
                            'hd'   [#patients,  #thresholds, nclasses]
                            'cardiac_phase': ES/ED
                            'cropped': 0=No, 1=Yes
                            'mc_dropout': False, True
    :param width:
    :param height:
    :param do_save:
    :param do_show:
    :param error_metric: 'dice', 'hd', 'seg_errors'
    :param output_dir
    :return:
    """

    assert error_metric in ['dice',  'hd', 'seg_errors']
    if do_save and output_dir is None:
        raise ValueError("ERROR - cannot save if output_dir is empty")

    axis_label_size = {'fontname': 'Monospace', 'size': '28', 'color': 'black', 'weight': 'normal'}

    tick_size = 20
    legend_size = 16
    linewidth = 2
    markersize = 10

    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()
    y_max = 0
    networks = []
    for idx, eval_obj_dict in enumerate(list_eval_obj):
        coverages = eval_obj_dict['coverages']  # eval_obj_dict['coverages']
        if not isinstance(eval_obj_dict, dict):
            eval_obj_dict = dict(eval_obj_dict)
        cardiac_phase = str(eval_obj_dict['cardiac_phase'])
        mc_dropout = bool(eval_obj_dict['mc_dropout'])
        type_of_map = "bayes" if mc_dropout else "entropy"
        network = eval_obj_dict['network']
        loss_func = eval_obj_dict['loss_function']
        if network not in networks:
            networks.append(network)

        if error_metric == 'seg_errors':
            y_values = np.mean(eval_obj_dict['seg_errors'], axis=0)  # average over patients
        elif error_metric == "dice":
            y_values = np.mean(1. - eval_obj_dict['dice'][..., 1:], axis=(0, 2))  # average over patients/classes
        else:
            y_values = np.mean(eval_obj_dict['hd'][..., 1:], axis=(0, 2))  # average over patients/classes
        if y_max < np.max(y_values):
            y_max = np.max(y_values)
        network_label = translate_combined_model_tag(network + "_" + loss_func)
        line_color, dash_style, alpha_value, marker_style = determine_plot_style(network, loss_func, mc_dropout)
        line_label = "{} {}".format(network_label, type_of_map)

        plt.plot((1 - coverages) * 100, y_values, marker=marker_style,
                 c=line_color, label=line_label, alpha=alpha_value, linewidth=linewidth,
                 linestyle='--', dashes=dash_style, markersize=markersize)
        # ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    plt.legend(loc="best", prop={'size': legend_size})
    plt.xlabel("Coverage (%)", **axis_label_size)
    if error_metric == 'seg_errors':
        plt.ylabel("#FP+FN", **axis_label_size)
        plt.ylim([0, y_max + (0.1 * y_max)])
    elif error_metric == "dice":
        plt.ylabel("1 - Dice ", **axis_label_size)
        plt.ylim([0, y_max + (0.1 * y_max)])
    elif error_metric == 'hd':
        plt.ylabel("HD ", **axis_label_size)
        plt.ylim([0, y_max + (0.1 * y_max)])
    # ax.yaxis.set_label_position("right")
    yticks = ax.yaxis.get_major_ticks()

    # ax.yaxis.tick_right()
    yticks[0].label2.set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.title("Coverage risk curve: {}".format(cardiac_phase), **axis_label_size)
    plt.xlim([0., 100])
    xvalues = np.arange(0, 110, 10)
    ax.set_xticks(xvalues)
    ax.set_xticklabels(xvalues[::-1]) # ['100', '95', '90', '80', '50', '30', '20', '10']
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    print("_".join(networks))
    if do_save:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        str_networks = "-".join(networks)
        suffix = "_" + eval_id if eval_id is not None else ""
        fig_name = "cov_risk_curve_" + cardiac_phase + "_" + error_metric + suffix
        fig_name = os.path.join(output_dir, fig_name + ".png")
        plt.savefig(fig_name, bbox_inches='tight')
        plt.savefig(fig_name.replace(".png", ".pdf"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_name))

    if do_show:
        plt.show()


def plot_coverage_risk_curve_per_loss(list_eval_obj, width=8, height=8, do_save=False, do_show=True,
                                      error_metric='dice', output_dir=None, eval_id=None):
    """

    :param list_eval_obj: list of numpy archive objects with the following dict entries:
                            'coverage' [#thresholds],
                            'dice' [#patients,  #thresholds, nclasses]
                            'hd'   [#patients,  #thresholds, nclasses]
                            'cardiac_phase': ES/ED
                            'cropped': 0=No, 1=Yes
                            'mc_dropout': False, True
    :param width:
    :param height:
    :param do_save:
    :param do_show:
    :param error_metric: 'dice', 'hd', 'seg_errors'
    :param output_dir
    :return:
    """

    assert error_metric in ['dice',  'hd', 'seg_errors']
    if do_save and output_dir is None:
        raise ValueError("ERROR - cannot save if output_dir is empty")

    axis_label_size = {'fontname': 'Monospace', 'size': '28', 'color': 'black', 'weight': 'normal'}
    small_title = {'fontname': 'Monospace', 'size': '18', 'color': 'black', 'weight': 'normal'}
    tick_size = 20
    legend_size = 20
    linewidth = 3.5
    markersize = 10

    fig = plt.figure(figsize=(width, height))
    ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2, sharey=ax1)
    pylab.setp(ax2.get_yticklabels(), visible=False)
    loss_functions = {1: ['CE', 'Brier'], 2: 'soft-Dice'}
    filter_lossfunction = [s.lower() for s in loss_functions[1]]
    # ax = plt.gca()
    y_max = 0
    networks = []
    for idx, eval_obj_dict in enumerate(list_eval_obj):
        if idx >= len(PLOT_COLORS):
            idx -= len(PLOT_COLORS)

        coverages = eval_obj_dict['coverages']  # eval_obj_dict['coverages']
        if not isinstance(eval_obj_dict, dict):
            eval_obj_dict = dict(eval_obj_dict)
        cardiac_phase = str(eval_obj_dict['cardiac_phase'])
        # fig.suptitle("{}".format(cardiac_phase), **axis_label_size)

        mc_dropout = bool(eval_obj_dict['mc_dropout'])
        type_of_map = "b-map" if mc_dropout else "e-map"
        network = eval_obj_dict['network']
        loss_func = eval_obj_dict['loss_function']
        network_label = translate_combined_model_tag(network + "_" + loss_func)
        if network not in networks:
            networks.append(network)

        if error_metric == 'seg_errors':
            y_values = np.mean(eval_obj_dict['seg_errors'], axis=0)  # average over patients
            y_std = np.std(eval_obj_dict['seg_errors'], axis=0)
        elif error_metric == "dice":
            y_values = np.mean(1. - eval_obj_dict['dice'][..., 1:], axis=(0, 2))  # average over patients/classes
            y_std = np.std(eval_obj_dict['dice'][..., 1:], axis=(0, 2))
        else:
            print(eval_obj_dict['hd'].shape)
            y_values = np.mean(eval_obj_dict['hd'][..., 1:], axis=(0, 2))  # average over patients/classes
            y_std = np.std(eval_obj_dict['hd'][..., 1:], axis=(0, 2))
        if y_max < np.max(y_values):
            y_max = np.max(y_values)

        # line_color, dash_style, alpha_value, marker_style = determine_plot_style(network, loss_func, mc_dropout)
        line_color = PLOT_COLORS[idx]
        alpha_value = ALPHA
        dash_style = LINESTYLES[network]
        marker_style = None
        line_label = "{} ({})".format(network_label, type_of_map)
        if loss_func in filter_lossfunction:
            print_ax = ax1
            # ax1.plot((1 - coverages) * 100, y_values, marker=marker_style,
            #          c=line_color, label=line_label, alpha=alpha_value, linewidth=linewidth,
            #          linestyle='--', dashes=dash_style, markersize=markersize)
        else:
            print_ax = ax2
        print_ax.plot((1 - coverages) * 100, y_values,  # marker=marker_style,
                 c=line_color, label=line_label, alpha=alpha_value, linewidth=linewidth,
                 linestyle=dash_style, markersize=markersize)  # , dashes=dash_style   linestyle='--'
        print_ax.fill_between((1 - coverages) * 100, y_values - y_std, y_values + y_std, alpha=0.08, color=line_color)
        # ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    ax1.legend(loc="best", prop={'size': legend_size})
    ax1.set_xlabel("Coverage (%)", **axis_label_size)
    ax2.legend(loc="best", prop={'size': legend_size})
    ax2.set_xlabel("Coverage (%)", **axis_label_size)
    if error_metric == 'seg_errors':
        ax1.set_ylabel("#FP+FN", **axis_label_size)
        ax1.set_ylim([0, y_max + (0.1 * y_max)])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylabel("#FP+FN", **axis_label_size)
        ax2.set_ylim([0, 3200])
    elif error_metric == "dice":
        ax1.set_ylabel("1 - Dice ", **axis_label_size)
        ax1.set_ylim([0, y_max + (0.1 * y_max)])
        # ax2.set_ylabel("1 - Dice ", **axis_label_size)
        ax2.set_ylim([0, y_max + (0.1 * y_max)])
    elif error_metric == 'hd':
        ax1.set_ylabel("HD ", **axis_label_size)
        ax1.set_ylim([0, y_max + (0.1 * y_max)])
        # ax2.set_ylabel("HD ", **axis_label_size)
        ax2.set_ylim([0, y_max + (0.1 * y_max)])
    # ax.yaxis.set_label_position("right")
    yticks = ax1.yaxis.get_major_ticks()

    # ax.yaxis.tick_right()
    yticks[0].label2.set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    ax1.set_title("{}".format(",".join(loss_functions[1])), **axis_label_size)  # **small_title)
    ax1.set_xlim([0., 100])
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)
    ax2.set_title("{}".format(loss_functions[2]), **axis_label_size)  # **small_title)
    ax2.set_xlim([0., 100])
    xvalues = np.arange(0, 110, 20)
    ax1.set_xticks(xvalues)
    ax1.set_xticklabels(xvalues[::-1]) # ['100', '95', '90', '80', '50', '30', '20', '10']
    ax2.set_xticks(xvalues)
    ax2.set_xticklabels(xvalues[::-1])
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    print("_".join(networks))
    if do_save:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        str_networks = "-".join(networks)
        suffix = "_" + eval_id if eval_id is not None else ""
        fig_name = "cov_risk_curve_" + cardiac_phase + "_" + error_metric + suffix
        fig_name = os.path.join(output_dir, fig_name + ".png")
        plt.savefig(fig_name, bbox_inches='tight')
        plt.savefig(fig_name.replace(".png", ".pdf"), bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_name))

    if do_show:
        plt.show()
    # plt.close()