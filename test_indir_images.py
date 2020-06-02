import argparse
import numpy as np
import os
from os import path
from tqdm import tqdm
import torch
from utils.trainers import get_trainer
from datasets.data_config import get_config
from datasets.ACDC.data import get_images_nifti, get_4dimages_nifti
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


def save_segmentations(segmentation, pat_id, output_dir, spacing, do_resample, new_spacing=None, frame_id=None,
                       direction=None, origin=None):

    if frame_id is not None:
        f_prefix = '{}_{:02d}'.format(pat_id, frame_id)
    else:
        f_prefix = '{}'.format(pat_id)
    fname = path.join(output_dir, f_prefix + '.nii.gz')
    print(segmentation.shape, fname)
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

    save_nifty(segmentation.astype(np.int16), new_spacing if do_resample else spacing, fname,
               direction=direction, origin=origin)


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a segmentation network')
    parser.add_argument('model_dir', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--output_directory', type=str, default=None, help='directory for experiment outputs')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--samples', type=int, default=1, help="If >1 we use MC-dropout during testing")
    parser.add_argument('--resample', action='store_true', help="Resample images to 1.4x1.4mm2")
    parser.add_argument('--patid', type=str, default=None, help="You can specify one patient id to process")

    # evaluate on different dataset than model was trained on
    parser.add_argument('--dataset', type=str, default=None, help='dataset, also used for setting input dir images to be segmented')
    args = parser.parse_args()
    args.model_dir = os.path.expanduser(args.model_dir)
    args.data_dir = os.path.expanduser(args.data_dir)
    if args.output_directory is None:
        args.output_directory = args.data_dir
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
    model_settings = loadExperimentSettings(path.join(args.model_dir, 'settings.yaml'))
    model_file = path.join(args.model_dir, str(model_settings.max_iters) + '.model')
    output_seg_dir = os.path.join(args.output_directory, "pred_labels")
    os.makedirs(output_seg_dir, exist_ok=True)
    # IMPORTANT: ASSUMING LAST LAYER HAS 4 OUTPUT CHANNELS and INPUT LAYER HAS ONE CHANNEL
    n_classes = 4
    n_channels_input = 1

    trainer, pad = get_trainer(model_settings, n_classes, n_channels_input, model_file=model_file)
    trainer.evaluate_with_dropout = use_mc
    print("INFO - arguments")
    print(args)
    testset_generator = get_images_nifti(args.data_dir, resample=False, patid=args.patid, rescale=True)
    pat_id_saved = None

    for sample in tqdm(testset_generator, desc="Generating segmentation volumes"):
        image, spacing, reference = sample['image'], sample['spacing'], sample['reference']
        pat_id, phase_id, frame_id = sample['patient_id'], sample['cardiac_phase'], sample['frame_id']
        num_of_frames = sample['num_of_frames']
        original_spacing = sample['original_spacing']
        shape_changed = False
        if pat_id_saved is None or pat_id != pat_id_saved:
            # initialize 4d segmentation volume
            segmentation4d = np.empty((0, image.shape[0], image.shape[1], image.shape[2]))

        if pad > 0 or model_settings.network[:4] == "unet" or model_settings.network[:3] == "drn":
            if model_settings.network[:4] == "unet":
                image, shape_changed, yx_padding = fit_unet_image(image, num_downsamplings=4)
            elif model_settings.network[:3] == "drn":
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
            do_resample = True if model_settings.resample or original_spacing[-1] < 1. else False
            # IMPORTANT: if frame_id is None (e.g. when processing 4D data) then filename is without suffix frame_id
            segmentation4d = np.squeeze(segmentation4d)
            save_segmentations(segmentation4d, pat_id, output_seg_dir, spacing, do_resample, new_spacing=original_spacing, frame_id=None)

        pat_id_saved = pat_id


if __name__ == '__main__':
    main()


# no_proxy=localhost CUDA_VISIBLE_DEVICES=7 python test_indir_images.py ~/expers/redo_expers/f0/drn_mc_ce/
#                   ~/expers/cardiac_sr/acdc/hcae_acai_new_combined_ps128_os8_4/images_sr/patient016/
#                   --output_directory=/home/jorg/bogus --resample --save_output
