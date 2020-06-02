import SimpleITK as sitk
import os.path as path
import os
import numpy as np
from datasets.ACDC.data import acdc_validation_fold, acdc_get_all_patients
from datasets.ARVC.dataset import arvc_get_evaluate_set
from datasets.common import apply_2d_zoom_3d


def save_nifty(image, spacing, fname, direction=None, origin=None):
    if isinstance(spacing, np.ndarray) and not spacing.dtype == np.float64:
        spacing = spacing.astype(np.float64)

    if image.ndim == 4:
        # 4d array
        if len(spacing) == 3:
            spacing = np.array([1, spacing[0], spacing[1], spacing[2]]).astype(np.float64)
        volumes = [sitk.GetImageFromArray(image[v], False) for v in range(image.shape[0])]
        image = sitk.JoinSeries(volumes)

    else:
        image = sitk.GetImageFromArray(image)

    if direction is not None:
        image.SetDirection(direction)
    if origin:
        image.SetOrigin(origin)

    image.SetSpacing(spacing[::-1])
    sitk.WriteImage(image, fname)


def get_test_set_generator(args, exp_settings, dta_settings, **kwargs):

    patid = kwargs.get('patid', None)
    all_frames = kwargs.get('all_frames', False)
    if exp_settings.dataset == "ACDC" and not args.acdc_all:
        test_set = acdc_validation_fold(exp_settings.fold,
                                                root_dir=dta_settings.short_axis_dir,
                                                limited_load=args.limited_load,
                                                resample=exp_settings.resample,
                                                patid=patid,
                                        resample_zaxis=args.super_resolution)
    elif exp_settings.dataset == "ARVC":
        test_set = arvc_get_evaluate_set("test", limited_load=args.limited_load, resample=exp_settings.resample,
                                         rescale=True, patid=patid, all_frames=all_frames)
    elif exp_settings.dataset == 'ACDC' and args.acdc_all:
        test_set = acdc_get_all_patients(root_dir=dta_settings.short_axis_dir,
                                                limited_load=args.limited_load,
                                                resample=exp_settings.resample,
                                                patid=patid,
                                                resample_zaxis=False)
    elif exp_settings.dataset == 'ACDC_full':
        # we trained on complete ACDC dataset and evaluate method on ARVC dataset (test)
        test_set = arvc_get_evaluate_set("test", limited_load=args.limited_load, resample=exp_settings.resample,
                                         rescale=True, patid=patid, all_frames=all_frames)

    return test_set


def save_pat_objects(pat_id, phase_id, segmentation, seg_errors, uncertainties, aleatoric,
                     type_of_map, spacing,
                     output_dirs, new_spacing=None, do_resample=False, direction=None, origin=None,
                     pred_probs=None):
    if do_resample and new_spacing is None:
        raise ValueError("ERROR - if resample tue then new_spacing cannot be None")
    if phase_id is not None:
        f_prefix = '{}_{}'.format(pat_id, phase_id)
    else:
        f_prefix = '{}'.format(pat_id)
    fname = path.join(output_dirs['pred_labels'], f_prefix + '.nii.gz')
    if do_resample:
        segmentation = apply_2d_zoom_3d(segmentation.astype(np.int16), spacing, order=0, do_blur=False,
                                        as_type=np.int16, new_spacing=new_spacing)
        # print("Resample segmentations ", o_shape, segmentation.shape)

    save_nifty(segmentation.astype(np.int16), new_spacing if do_resample else spacing, fname,
               direction=direction, origin=origin)
    print("INFO - Saved auto mask to {} (sp/new_sp)".format(fname), spacing, new_spacing)
    fname = path.join(output_dirs['umaps'], f_prefix + '_{}.nii.gz'.format(type_of_map))
    if do_resample:
        uncertainties = apply_2d_zoom_3d(uncertainties.astype(np.float32), spacing, do_blur=True,
                                         new_spacing=new_spacing)
        # print("Resample uncertainties ", o_shape, uncertainties.shape)
    save_nifty(uncertainties.astype(np.float32), new_spacing if do_resample else spacing, fname, direction=direction,
               origin=origin)
    if seg_errors is not None:
        fname = path.join(output_dirs['errors'], f_prefix + '.nii.gz')
        if do_resample:
            seg_errors = apply_2d_zoom_3d(seg_errors.astype(np.int16), spacing, order=0, do_blur=False,
                                          as_type=np.int16, new_spacing=new_spacing)
        save_nifty(seg_errors.astype(np.int16), new_spacing if do_resample else spacing, fname, direction=direction,
                   origin=origin)
    if aleatoric is not None:
        fname = path.join(output_dirs['umaps'], f_prefix + '_aleatoric.nii.gz')
        if do_resample:
            aleatoric = apply_2d_zoom_3d(aleatoric.astype(np.float32), spacing, do_blur=True, new_spacing=new_spacing)
        save_nifty(aleatoric.astype(np.float32), new_spacing if do_resample else spacing, fname, direction=direction,
                   origin=origin)

    if pred_probs is not None:
        fname = path.join(output_dirs['pred_probs'], f_prefix + '.nii.gz')
        pred_new_spacing = np.array([new_spacing[0], 1, new_spacing[1], new_spacing[2]])
        pred_spacing = np.array([spacing[0], 1, spacing[1], spacing[2]])
        if do_resample:
            nclasses = pred_probs.shape[1]
            resampled_probs = np.zeros((segmentation.shape[0], pred_probs.shape[1], segmentation.shape[1],
                                        segmentation.shape[2]))
            for cls_idx in np.arange(nclasses):
                resampled_probs[:, cls_idx] = apply_2d_zoom_3d(pred_probs[:, cls_idx].astype(np.float32), spacing,
                                                               do_blur=False, new_spacing=new_spacing)
            pred_probs = resampled_probs
        print("WARNING do resample ", do_resample)
        save_nifty(pred_probs.astype(np.float32), pred_new_spacing if do_resample else pred_spacing, fname, direction=direction,
                   origin=origin)


def save_pred_probs(pat_id, phase_id, pred_probs, spacing,
                    output_dirs, new_spacing=None, do_resample=False, direction=None, origin=None):
    if phase_id is not None:
        f_prefix = '{}_{}'.format(pat_id, phase_id)
    else:
        f_prefix = '{}'.format(pat_id)

    fname = path.join(output_dirs['pred_probs'], f_prefix + '.nii.gz')
    pred_new_spacing = np.array([new_spacing[0], 1, new_spacing[1], new_spacing[2]])
    pred_spacing = np.array([spacing[0], 1, spacing[1], spacing[2]])
    if do_resample:
        nclasses = pred_probs.shape[1]
        resampled_probs = None
        for cls_idx in np.arange(nclasses):
            new_probs = apply_2d_zoom_3d(pred_probs[:, cls_idx].astype(np.float32), spacing,
                                                           do_blur=False, new_spacing=new_spacing)
            if resampled_probs is None:
                resampled_probs = np.zeros((new_probs.shape[0], pred_probs.shape[1], new_probs.shape[1],
                                            new_probs.shape[2]))
            resampled_probs[:, cls_idx] = new_probs
        pred_probs = resampled_probs

    save_nifty(pred_probs.astype(np.float32), pred_new_spacing if do_resample else pred_spacing, fname, direction=direction,
               origin=origin)


def make_dirs(output_dir, use_mc):
    dir_suffix = "_mc" if use_mc else ""
    output_dirs = {'pred_labels': path.join(output_dir, "pred_labels" + dir_suffix)}
    if not path.isdir(output_dirs['pred_labels']):
        os.makedirs(output_dirs['pred_labels'], exist_ok=True)
    output_dirs['umaps'] = path.join(output_dir, "umaps")
    if not os.path.isdir(output_dirs['umaps']):
        os.makedirs(output_dirs['umaps'], exist_ok=True)
    output_dirs['errors'] = path.join(output_dir, "errors" + dir_suffix)
    if not os.path.isdir(output_dirs['errors']):
        os.makedirs(output_dirs['errors'], exist_ok=True)
    output_dirs['pred_probs'] = path.join(output_dir, "pred_probs" + dir_suffix)
    if not os.path.isdir(output_dirs['pred_probs']):
        os.makedirs(output_dirs['pred_probs'], exist_ok=True)

    return output_dirs
