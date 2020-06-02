
import argparse
import numpy as np
import os
from os import path
from tqdm import tqdm
import torch
from utils.trainers import get_trainer
from datasets.data_config import get_config
from datasets.ACDC.data import acdc_validation_fold_image4d, get_4dimages_nifti
from datasets.common import apply_2d_zoom_3d
from evaluate.acdc_patient_eval import VolumeEvaluation
from evaluate.predictions import Predictions
from evaluate.common import save_nifty, make_dirs
from utils.common import loadExperimentSettings


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


def restore_original_size(output, yx_padding):
    """

    :param output: dictionary with 2 or 3 pytorch tensors:
                    'softmax', 'predictions' and optional 'aleatoric'
            all output tensors have shape [z, 1, y, x]

    :param yx_padding:
    :return:
    """
    pady_0, pady_1, padx_0, padx_1 = yx_padding
    _, _, y_size, x_size = output["softmax"].shape
    output["softmax"] = output["softmax"][..., pady_0:y_size-pady_1, padx_0:x_size-padx_1]
    output["predictions"] = output["predictions"][..., pady_0:y_size-pady_1, padx_0:x_size-padx_1]
    if 'aleatoric' in output.keys():
        output['aleatoric'] = output['aleatoric'][..., pady_0:y_size-pady_1, padx_0:x_size-padx_1]
    return output


def fit_unet_image(image, num_downsamplings=4):
    # assuming image is numpy array of size [z, y, x]
    org_shape = image.shape

    def determine_pad_size(dim_size, denom=16):
        # for UNet we have to make sure that x and y size are four times dividable by 2 (hence 8)
        d = dim_size / denom

        if d % 2 != 0:
            # if decimal part we ceil otherwise we add one
            new_size = np.ceil(d) * denom if d % 1 != 0 else (d + 1) * denom
            diff = new_size - dim_size
            if diff % 2 == 0:
                return int(diff / 2), int(diff / 2)
            else:
                return 0, int(diff)

        else:
            return 0, 0

    _, y_org, x_org = org_shape
    padx_0, padx_1 = determine_pad_size(x_org, denom=2**num_downsamplings)
    pady_0, pady_1 = determine_pad_size(y_org, denom=2**num_downsamplings)
    image = np.pad(image, ((0, 0), (pady_0, pady_1), (padx_0, padx_1)), mode="edge")
    return image, org_shape != image.shape, tuple((pady_0, pady_1, padx_0, padx_1))


def save_segmentations(segmentation, pat_id, output_dirs, spacing, do_resample, new_spacing=None, frame_id=None,
                       direction=None, origin=None):

    if frame_id is not None:
        f_prefix = '{}_{:02d}'.format(pat_id, frame_id)
    else:
        f_prefix = '{}'.format(pat_id)
    fname = path.join(output_dirs['pred_labels'], f_prefix + '.nii.gz')
    if do_resample:
        if segmentation.ndim == 3:
            segmentation = apply_2d_zoom_3d(segmentation.astype(np.int16), spacing, order=0, do_blur=False,
                                            as_type=np.int16, new_spacing=new_spacing)
        else:
            # assuming 4d data
            num_time_points = segmentation.shape[0]
            resized_segs = None
            for t in np.arange(num_time_points):
                seg = segmentation[t]
                new_seg = apply_2d_zoom_3d(seg.astype(np.int16), spacing, order=0, do_blur=False,
                                            as_type=np.int16, new_spacing=new_spacing)
                if resized_segs is None:
                    resized_segs = np.expand_dims(new_seg, axis=0)
                else:
                    resized_segs = np.vstack((resized_segs, np.expand_dims(new_seg, axis=0)))

        # print("Resample segmentations ", o_shape, segmentation.shape)

    save_nifty(segmentation.astype(np.int16), new_spacing if do_resample else spacing, fname,
               direction=direction, origin=origin)


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a segmentation network')
    parser.add_argument('experiment_directory', type=str)
    parser.add_argument('--output_directory', type=str, default=None, help='directory for experiment outputs')
    parser.add_argument('--limited_load', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--samples', type=int, default=1, help="If >1 we use MC-dropout during testing")
    parser.add_argument('--checkpoint', type=int, default=100000, help="Checkpoint for testing")
    parser.add_argument('--resample', action='store_true', help="Resample images to 1.4x1.4mm2")
    parser.add_argument('--save_probs', action='store_true', help="Save softmax probabilities")
    parser.add_argument('--patid', type=str, default=None, help="You can specify one patient id to process")
    parser.add_argument('--rescale', action='store_true', help="Test with super resolution of images")
    parser.add_argument('--acdc_all', action='store_true', help="Evaluate on all ACDC patient studies")
    # evaluate on different dataset than model was trained on
    parser.add_argument('--dataset', type=str, default=None, help='dataset, also used for setting input dir images to be segmented')
    args = parser.parse_args()
    if args.output_directory is None:
        args.output_directory = args.experiment_directory
    args.output_directory = os.path.expanduser(args.output_directory)
    return args


def main():
    # first we obtain the user arguments, set random seeds, make directories, and store the experiment settings.
    args = parse_args()
    if args.samples > 1:
        use_mc = True
    else:
        use_mc = False

    os.makedirs(args.output_directory, exist_ok=True)
    experiment_settings = loadExperimentSettings(path.join(args.experiment_directory, 'settings.yaml'))
    # if we pass args.dataset (to evaluate on dataset different than trained on) use it to set input dir for images to be segmented
    dta_settings = get_config(dataset=args.dataset if args.dataset is not None else experiment_settings.dataset)

    model_file = path.join(args.experiment_directory, str(args.checkpoint) + '.model')
    output_dirs = make_dirs(args.output_directory, use_mc)
    # we create a trainer
    if experiment_settings.dataset != dta_settings.dataset:
        n_classes = len(get_config(experiment_settings.dataset).tissue_structure_labels)
    else:
        n_classes = len(dta_settings.tissue_structure_labels)
    n_channels_input = 1

    trainer, pad = get_trainer(experiment_settings, n_classes, n_channels_input, model_file=model_file)
    trainer.evaluate_with_dropout = use_mc
    print("WARNING - Rescaling intensities is set to {}".format(args.rescale))
    if args.dataset is None:
        # TODO TEMPORARY !!!!
        root_dir = os.path.expanduser("~/data/ACDC_SR/")
        testset_generator = acdc_validation_fold_image4d(experiment_settings.fold,
                                                         root_dir=root_dir,  # dta_settings.short_axis_dir,
                                                         file_suffix="4d_acai.nii.gz",
                                                         resample=experiment_settings.resample,
                                                         patid=args.patid,
                                                         rescale=args.rescale)
    else:
        print("INFO - You passed following arguments")
        print(args)
        testset_generator = get_4dimages_nifti(dta_settings.short_axis_dir, resample=False, patid=args.patid, rescale=True)
    pat_id_saved = None

    for sample in tqdm(testset_generator, desc="Generating 4d segmentation volumes"):
        image, spacing, reference = sample['image'], sample['spacing'], sample['reference']
        pat_id, phase_id, frame_id = sample['patient_id'], sample['cardiac_phase'], sample['frame_id']
        num_of_frames = sample['num_of_frames']
        original_spacing = sample['original_spacing']
        shape_changed = False
        if pat_id_saved is None or pat_id != pat_id_saved:
            # initialize 4d segmentation volume
            segmentation4d = np.empty((0, image.shape[0], image.shape[1], image.shape[2]))

        if pad > 0 or experiment_settings.network[:4] == "unet" or experiment_settings.network[:3] == "drn":
            if experiment_settings.network[:4] == "unet":
                image, shape_changed, yx_padding = fit_unet_image(image, num_downsamplings=4)
            elif experiment_settings.network[:3] == "drn":
                # print("WARNING - adjust image", image.shape)
                image, shape_changed, yx_padding = fit_unet_image(image, num_downsamplings=3)
            else:
                # image has shape [z, y, x] so pad last two dimensions
                image = np.pad(image, ((0,0), (pad, pad), (pad, pad)), mode="edge") # "'constant', constant_values=(0,))

        image = image[:, None]  # add extra dim of size 1
        image = torch.from_numpy(image)
        pat_predictions = Predictions()
        with torch.set_grad_enabled(False):
            for s in range(args.samples):
                output = trainer.predict(image)
                if shape_changed:

                    output = restore_original_size(output, yx_padding)
                    # print("WARNING - restore original size", output["softmax"].shape)
                soft_probs = output['softmax'].detach().numpy()
                pat_predictions(soft_probs, cardiac_phase_tag=phase_id, pred_logits=None)
                # torch.cuda.empty_cache()

        # if mc_dropout is true we compute the Bayesian maps (stddev) otherwise Entropy. Method makes sure
        # that in case we're sampling that pred_probs are averaged over samples.
        # 14-8-2019 IMPORTANT: We are computing Bayesian maps with MEAN stddev over classes! Before MAX
        pred_probs, uncertainties = pat_predictions.get_predictions(compute_uncertainty=True, mc_dropout=use_mc,
                                                                    agg_func="mean")
        segmentation = np.argmax(pred_probs, axis=1)
        eval_obj = VolumeEvaluation(pat_id, segmentation, reference,
                                    voxel_spacing=spacing, num_of_classes=n_classes,
                                    mc_dropout=use_mc, cardiac_phase=phase_id)

        eval_obj.post_processing_only()
        # IMPORTANT: in fast_evaluate we post process the predictions only keeping the largest connected components
        segmentation = eval_obj.pred_labels
        # print(segmentation4d.shape, segmentation.shape, segmentation[None].shape, spacing, original_spacing)
        segmentation4d = np.vstack((segmentation4d, segmentation[None]))
        del output

        if args.save_output and num_of_frames == frame_id + 1:
            do_resample = True if experiment_settings.resample or original_spacing[-1] < 1. else False
            # IMPORTANT: if frame_id is None (e.g. when processing 4D data) then filename is without suffix frame_id
            save_segmentations(segmentation4d, pat_id, output_dirs, spacing, do_resample, new_spacing=original_spacing, frame_id=None)

        pat_id_saved = pat_id


if __name__ == '__main__':
    main()


# no_proxy=localhost CUDA_VISIBLE_DEVICES=7 python test_single_4d_volume.py ~/expers/acdc_full/drn_ce --dataset=vumc_pulmonary --samples=1 --save_output