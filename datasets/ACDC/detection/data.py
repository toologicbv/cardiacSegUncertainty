import numpy as np
import os
from datasets.ACDC.get_data import load_data
from datasets.data_config import get_config
from datasets.ACDC.detection.distance_transforms import generate_adjusted_dt_map, FilterSegErrors

DTA_SETTINGS = get_config('ACDC')


def generate_detector_labels(dt_map, seg_errors, apex_base_slices, fname):
    """
    Generation of target areas in automatic segmentations (test set) that we want to inspect after prediction.
    These areas must be detected by our "detection model". So there're just for supervised learning
    For each patient study we produce numpy array with shape [num_of_classes, w, h, #slices]

    IMPORTANT: we have different rois for the automatic segmentations produced by single prediction (dropout=False)
               or by a Bayesian network using T (we used 10) samples. In the latter case mc_dropout=True.

    :param dt_map:
    :param seg_errors:
    :param apex_base_slices: dict 'A': scalar, 'B': scalar
    :param fname: absolute file name
    :return:
    """

    segfilter_handler = FilterSegErrors(seg_errors, dt_map, apex_base_slices)
    # disabled 25-3: self.detector_labels[patient_id][frame_id] = determine_target_voxels(auto_pred, labels, dt_map)
    dt_labels = segfilter_handler.get_filtered_errors()
    np.savez(fname, dt_ref_labels=dt_labels)
    return dt_labels


def detection_generator(src_data_path, error_margins, patid=None, do_save_nifti=False,
                        mc_dropout=False, cardiac_phase=None,
                        use_existing_dt_maps=True):
    """
    Generate distance transform maps, then target labels for supervised training of detector
    and finally save the last as Nifti images for visualization

    :param src_data_path: path to experiment directory or directory where pred_labels, umaps dir are located
    :param error_margins: tuple (outside margin, inside margin) w.r.t. contour surface.
                                 both in mm.
                                 e.g. 4.2mm and 2.8mm. We tolerate less distance of errors to surface
                                 for seg errors located inside of contour object
    :param patid:
    :param do_save_nifti:
    :param mc_dropout:
    :param use_existing_dt_maps: boolean, re-use existing dt maps, only generate target labels again
    :param error_margins: IMPORTANT: in order to test different runs with fixed margins
                tuple of 2 numbers: (1) outside margin (mm) (2) inside margin (mm)
    :return:
    """

    def init_dirs(root_dir, dt_config_id):
        dt_map_dir = os.path.join(root_dir, "dt_maps" + os.sep + dt_config_id)
        if not os.path.isdir(dt_map_dir):
            os.makedirs(dt_map_dir, exist_ok=False)
        dt_label_dir = os.path.join(root_dir, "dt_labels" + os.sep + dt_config_id)
        if not os.path.isdir(dt_label_dir):
            os.makedirs(dt_label_dir, exist_ok=False)
        return dt_map_dir, dt_label_dir

    mc_suffix = "_mc" if mc_dropout else ""
    dt_config_id = "fixed_" + str(error_margins[0]).replace(".", "") + "_" \
                   + str(error_margins[1]).replace(".", "")
    dt_map_dir, dt_label_dir = init_dirs(src_data_path, dt_config_id)
    print("WARNING - using fixed margins outside: "
          "{:.3f} inside: {:.3f}".format(error_margins[0],
                                         error_margins[1]))
    print("WARNING - using {} as inut/output dir for dt-maps!".format(dt_map_dir))

    if cardiac_phase is None:
        cardiac_phases = ['ED', 'ES']
    else:
        cardiac_phases = [cardiac_phase]
    for cardiac_phase in cardiac_phases:
        if use_existing_dt_maps:
            data_types = ['pred_labels', 'ref_labels', 'dt_maps']
        else:
            data_types = ['pred_labels', 'ref_labels']
        data_dict = load_data(src_data_path, cardiac_phase, data_types, mc_dropout, dt_config_id=dt_config_id,
                              meta_info=True)

        avg_spacing = np.zeros(3)
        if patid is None:
            patient_ids = data_dict['pred_labels'].keys()
        else:
            patient_ids = [patid]

        for patid in patient_ids:
            ref_labels = data_dict['ref_labels'][patid]
            # because we set meta_info = True in function call laod_data we're dealing with a dict per patient id
            # we do this to obtain the spacing which we need for the computation of the distance maps
            pred_labels = data_dict['pred_labels'][patid]['pred_labels']
            spacing = data_dict['pred_labels'][patid]['spacing']  # shape [z, y, x] where y=x always
            avg_spacing += spacing
            apex_base_slices = data_dict['apex_base_slices'][patid]  # dict keys 'A': scalar, 'B': scalar
            if use_existing_dt_maps:
                dt_map = data_dict['dt_maps'][patid]
            else:
                # use voxelspacing from this image. because spacing in-plane is isotropic we can use x or y spacing
                dt_map = generate_adjusted_dt_map(ref_labels, error_margins,
                                                  voxelspacing=spacing[1])
                # REMEMBER: dt maps are independent of whether we're dealing with mc dropout or not.
                # so basically if they already exist we overwrite them here
                fname = os.path.join(dt_map_dir, patid + "_{}".format(cardiac_phase) + ".npz")
                np.savez(fname, dt_map=dt_map, error_margins=error_margins)

            seg_errors = np.zeros_like(ref_labels)
            # Note: all objects have shape [z, nclasses, y, x] we skip the bg class for seg errors
            # print(patid, ref_labels.shape, pred_labels.shape)
            seg_errors[:, 1:] = (ref_labels[:, 1:] != pred_labels[:, 1:]).astype(np.int)
            fname = os.path.join(dt_label_dir, patid + "_{}".format(cardiac_phase) + mc_suffix + ".npz")
            dt_ref_labels = generate_detector_labels(dt_map, seg_errors, apex_base_slices, fname)
            if do_save_nifti:
                # target_labels has shape [nclasses, x, y, z]
                mask = np.zeros(dt_ref_labels.shape[1:])
                # create binary mask with all errors that need to be detected
                mask[np.sum(dt_ref_labels, axis=0) != 0] = 1
    print("Avg spacing ", avg_spacing * 1/len(patient_ids))


if __name__ == '__main__':
    # usage
    # CUDA_VISIBLE_DEVICES=2 python datasets/ACDC/detection/data.py ~/expers/acdc/f3/drn_mc_ce/
    import argparse

    parser = argparse.ArgumentParser(description='generate distance transform maps & detection labels')
    parser.add_argument('input_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('--margins', type=float, nargs=2, default=DTA_SETTINGS.dt_margins)
    parser.add_argument('--reuse_dt_maps', action='store_true')
    parser.add_argument('--patid', type=str, default=None)
    args = parser.parse_args()
    src_data_path = os.path.expanduser(args.input_directory)
    error_margins = args.margins
    cardiac_phases = ["ED", "ES"]
    use_dropout = [False, True]

    for cardiac_phase in cardiac_phases:
        for mc_dropout in use_dropout:
            print("INFO - Generate dt-maps and labels for {}/{} (margins {} {})".format(cardiac_phase, mc_dropout,
                                                                                        error_margins[0],
                                                                                        error_margins[1]))
            detection_generator(src_data_path, error_margins, patid=args.patid, do_save_nifti=False,
                                mc_dropout=mc_dropout, cardiac_phase=cardiac_phase,
                                use_existing_dt_maps=args.reuse_dt_maps)

