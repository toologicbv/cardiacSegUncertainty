
import argparse
import numpy as np
import os
from os import path
import torch
from utils.trainers import get_trainer
from datasets.data_config import get_config
from evaluate.acdc_patient_eval import VolumeEvaluation
from evaluate.test_results import ACDCTestResult
from evaluate.predictions import Predictions
from evaluate.common import get_test_set_generator, save_pat_objects, make_dirs, save_pred_probs
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
    parser.add_argument('--super_resolution', action='store_true', help="Test with super resolution of images")
    parser.add_argument('--acdc_all', action='store_true', help="Evaluate on all ACDC patient studies")
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
        type_of_map = "bmap"
        res_suffix = "_mc.npz"
    else:
        use_mc = False
        type_of_map = "emap"
        res_suffix = ".npz"

    print("INFO - Evaluating with super resolution = {}".format(args.super_resolution))
    os.makedirs(args.output_directory, exist_ok=True)
    experiment_settings = loadExperimentSettings(path.join(args.experiment_directory, 'settings.yaml'))
    dta_settings = get_config(experiment_settings.dataset)

    model_file = path.join(args.experiment_directory, str(args.checkpoint) + '.model')
    output_dirs = make_dirs(args.output_directory, use_mc)
    # we create a trainer
    n_classes = len(dta_settings.tissue_structure_labels)
    n_channels_input = 1

    trainer, pad = get_trainer(experiment_settings, n_classes, n_channels_input, model_file=model_file)
    trainer.evaluate_with_dropout = use_mc
    test_results = ACDCTestResult()
    testset_generator = get_test_set_generator(args, experiment_settings, dta_settings, patid=args.patid)

    for sample in testset_generator:
        image, spacing, reference = sample['image'], sample['spacing'], sample['reference']
        pat_id, phase_id, frame_id = sample['patient_id'], sample['cardiac_phase'], sample['frame_id']
        original_spacing = sample['original_spacing']
        shape_changed = False
        # if pat_id == "patient037" and phase_id == "ES":
        #     print("WARNING - Skip {}".format(pat_id))
        #     continue
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
                aleatoric = None if 'aleatoric' not in output.keys() else np.squeeze(output['aleatoric'])
                aleatoric = None if use_mc else aleatoric
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

        eval_obj.fast_evaluate(compute_hd=True)
        # IMPORTANT: in fast_evaluate we post process the predictions only keeping the largest connected components
        segmentation = eval_obj.pred_labels
        test_results(eval_obj.dice, hd=eval_obj.hd, cardiac_phase_tag=phase_id, pat_id=pat_id, hd95=eval_obj.hd95,
                     assd=eval_obj.assd)
        eval_obj.show_results()
        if args.save_output:
            # print("INFO - image/reference size ", image.shape, reference.shape)
            do_resample = True if experiment_settings.resample or original_spacing[-1] < 1. else False
            save_pat_objects(pat_id, phase_id, segmentation, None, uncertainties, aleatoric, type_of_map,
                             spacing, output_dirs, new_spacing=original_spacing, do_resample=do_resample,
                             pred_probs=pred_probs if args.save_probs else None)
        # Work-around to save predicted probabilities only
        if args.save_probs and not args.save_output:
            do_resample = True if experiment_settings.resample or original_spacing[-1] < 1. else False
            save_pred_probs(pat_id, phase_id, pred_probs, spacing, output_dirs
                            , new_spacing=original_spacing, do_resample=do_resample,
                            direction=None, origin=None)

        del output

    test_results.show_results()
    test_results.excel_string()

    if args.save_results:
        fname = path.join(args.output_directory,
                          "results_f" + str(experiment_settings.fold) + "_{}".format(len(test_results.pat_ids)) +
                          res_suffix)
        test_results.save(filename=fname)
        print("INFO - performance results saved to {}".format(fname))


if __name__ == '__main__':
    main()
