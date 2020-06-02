import glob
from shutil import copy2
import os
import argparse
import numpy as np

"""
    ["pred_labels", "dt_labels", "umaps", "ref_labels"]

"""


def get_test_patient_nums(fold):
    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)
    return allpatnumbers[foldmask == fold]


def move_to_model_root_dir(model_name, root_dir, margins="fixed_46_31"):
    # root_dir: e.g. ~/expers/redo_expers/
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    target_dir_root = os.path.join(root_dir, model_name)
    if not os.path.isdir(target_dir_root):
        os.makedirs(target_dir_root)

    for fold in range(4):
        print("---------------------------- Fold {} ----------------------------".format(fold))
        source_dir_fold = os.path.join(root_dir, "f{}".format(fold) + os.sep + model_name)

        test_patids = get_test_patient_nums(fold)
        for i, int_patid in enumerate(test_patids):
            for mc_dropout in [False, True]:
                for cardiac_phase in ['ED', 'ES']:
                    str_patid = "patient{:03}_{}".format(int_patid, cardiac_phase)
                    if mc_dropout:
                        str_patid = str_patid + "_mc"
                    if mc_dropout:
                        dir_suffix_pred_labels = "pred_labels_mc" + os.sep + str_patid.replace("_mc", "") + ".nii.gz"
                        target_pred_labels_dir = os.path.join(target_dir_root, "pred_labels_mc")
                        dir_suffix_umaps = "umaps" + os.sep + str_patid.replace("_mc", "") + "_bmap.nii.gz"
                        target_umaps_dir = os.path.join(target_dir_root, "umaps")
                    else:
                        dir_suffix_pred_labels = "pred_labels" + os.sep + str_patid.replace("_mc", "") + ".nii.gz"
                        target_pred_labels_dir = os.path.join(target_dir_root, "pred_labels")
                        dir_suffix_umaps = "umaps" + os.sep + str_patid.replace("_mc", "") + "_emap.nii.gz"
                        target_umaps_dir = os.path.join(target_dir_root, "umaps")
                    # for detection labels only difference between mc_dropout True/False is _mc suffix patient
                    src_dt_labels_file = os.path.join(source_dir_fold, "dt_labels" + os.sep + margins + os.sep + str_patid + ".npz")
                    target_dt_labels_dir = os.path.join(target_dir_root, "dt_labels" + os.sep + margins)
                    src_pred_labels_file = os.path.join(source_dir_fold, dir_suffix_pred_labels)
                    src_umaps_file = os.path.join(source_dir_fold, dir_suffix_umaps)
                    if not os.path.isdir(target_pred_labels_dir):
                        os.makedirs(target_pred_labels_dir)
                    if not os.path.isdir(target_umaps_dir):
                        os.makedirs(target_umaps_dir)
                    if not os.path.isdir(target_dt_labels_dir):
                        os.makedirs(target_dt_labels_dir)

                    # pred labels
                    copy2(src_pred_labels_file, target_pred_labels_dir)
                    # umaps
                    copy2(src_umaps_file, target_umaps_dir)
                    # dt_labels
                    copy2(src_dt_labels_file, target_dt_labels_dir)
        print("INFO - Copied files for {} patients in fold {}".format(i+1, fold))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='move files to prepare simulation of detected segmentation failures')
    parser.add_argument('src_path_data', type=str, help='directory that contains data objects from step 1')  # e.g. ~/expers/redo_expers/
    parser.add_argument('--seg_model', type=str, default=None)
    parser.add_argument('--loss_function', type=str, default=None)
    parser.add_argument('--margins', type=str, default="fixed_46_31")
    args = parser.parse_args()
    model_name = args.seg_model + "_" + args.loss_function
    root_target_dir = os.path.expanduser(args.src_path_data)  # e.g. ~/expers/redo_expers/

    move_to_model_root_dir(model_name, root_target_dir, args.margins)
