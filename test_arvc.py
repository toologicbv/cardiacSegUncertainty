
import argparse
import numpy as np
import os
from os import path
from evaluate.common import get_test_set_generator, save_pat_objects, make_dirs
import torch
from utils.trainers import get_trainer
from datasets.data_config import get_config
from evaluate.acdc_patient_eval import VolumeEvaluation
from evaluate.test_results import ARVCTestResult
from evaluate.predictions import Predictions
from utils.losses import get_loss_mask
from utils.common import loadExperimentSettings


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a segmentation network')
    parser.add_argument('experiment_directory', type=str)
    parser.add_argument('--output_directory', type=str, help='directory for experiment outputs', default=None)
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('--limited_load', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--samples', type=int, default=1, help="If >1 we use MC-dropout during testing")
    parser.add_argument('--checkpoint', type=int, default=100000, help="Checkpoint for testing")
    parser.add_argument('--resample', action='store_true', help="Resample images to 1.4x1.4mm2")
    parser.add_argument('--patid', type=str, default=None, help="You can specify one patient id to process")
    parser.add_argument('--all_frames', action='store_true', help="segment complete cardiac cycle")
    args = parser.parse_args()
    if args.output_directory is None:
        args.output_directory = args.experiment_directory

    return args


def check_save(pat_saved, sample_saved, result_obj, args, experiment_settings, output_dirs, type_of_map):

    if args.save_output and pat_saved is not None:
        # print(pat_saved, pat_id, sample_saved['spacing'])
        # segmentation = result_obj['segmentation']
        # print(segmentation.shape, np.count_nonzero(segmentation), np.unique(segmentation))
        save_pat_objects(pat_saved, None, result_obj['segmentation'], None,
                         result_obj['uncertainty'], result_obj['aleatoric'], type_of_map,
                         sample_saved['spacing'], output_dirs, new_spacing=sample_saved['original_spacing'],
                         do_resample=experiment_settings.resample, direction=sample_saved['direction'],
                         origin=sample_saved['origin'])


def prepare_transfer_learning(n_classes, ref_labels, cls_trans_dict, ignore_labels=None):

    if ignore_labels is not None and np.count_nonzero(ignore_labels) != 0:
        new_ignore_labels = np.zeros(n_classes).astype(ignore_labels.dtype)
        # translate ignore labels is binary vector of length n_classes (target dataset, ARVC)
        # remember: ignore_labels is transformed into multi-label vector with np.nonzero
        for c in np.nonzero(ignore_labels)[0]:
            new_ignore_labels[cls_trans_dict[c]] = 1
        ignore_labels = new_ignore_labels

    # translate multi-class labels from ARVC (target) to source (ACDC)
    new_ref_labels = np.zeros_like(ref_labels).astype(ref_labels.dtype)
    for c_tgt, c_src in cls_trans_dict.items():
        new_ref_labels[ref_labels == c_tgt] = c_src
    return new_ref_labels, ignore_labels


def prepare_result_obj(pat_saved, sample_new, result_obj, experiment_settings):

    if pat_saved != sample_new['patient_id']:
        o_shape = tuple((sample_new['number_of_frames'],)) + sample_new['image'].shape
        result_obj = {'segmentation': np.zeros(o_shape), 'uncertainty': np.zeros(o_shape),
                      'aleatoric': np.zeros(o_shape) if 'mcc' in experiment_settings.network else None}

    return result_obj


def process_volume(result_obj, sample, segmentation, uncertainty, aleatoric=None):
    frame_id = sample['frame_id']

    result_obj['segmentation'][frame_id] = segmentation
    result_obj['uncertainty'][frame_id] = uncertainty
    if aleatoric is not None:
        result_obj['aleatoric'][frame_id] = aleatoric

    return result_obj


def prepare_evaluation(sample, pat_saved, c_phase_saved, phase_results):

    ignore_labels = None if 'ignore_label' not in sample.keys() else sample['ignore_label']
    merge_results = 0
    if np.count_nonzero(ignore_labels) != 0:
        if pat_saved != sample['patient_id']:
            # new patient and labels are spread over different phases
            phase_results = dict()
            merge_results = 1
        elif c_phase_saved != sample['cardiac_phase']:
            # same patient but different phase
            phase_results = dict()
            merge_results = 1
        elif c_phase_saved == sample['cardiac_phase']:
            # phase_results contains already the previous evaluation results of the same c-phase for one tissue class
            # we will merge these results at the end of this iteration (see process_volume results)
            merge_results = 2
        else:
            pass
    else:
        phase_results = None

    return ignore_labels, merge_results, phase_results


def process_volume_results(test_results, eval_obj, pat_id, phase_id, merge_results, phase_results=None):
    if merge_results == 0:
        # Volume contains all tissue structures, just add result to test result object
        test_results(eval_obj.dice, hd=eval_obj.hd, cardiac_phase_tag=phase_id, pat_id=pat_id)
    elif merge_results == 1:
        # We need to merge the current results LATER with the second volume we evaluate for the OTHER phase
        phase_results['dice'], phase_results['hd'] = eval_obj.dice, eval_obj.hd
    elif merge_results == 2:
        # merge previous results into current eval object
        for cls_idx in eval_obj.class_indices:
            phase_results['dice'][cls_idx] = eval_obj.dice[cls_idx]
            phase_results['hd'][cls_idx] = eval_obj.hd[cls_idx]
        # print("Merge")
        # print(phase_results['dice'])
        # print(phase_results['hd'])
        test_results(phase_results['dice'], hd=phase_results['hd'], cardiac_phase_tag=phase_id, pat_id=pat_id)

    return phase_results


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

    os.makedirs(args.output_directory, exist_ok=True)
    experiment_settings = loadExperimentSettings(path.join(args.experiment_directory, 'settings.yaml'))
    dta_settings = get_config("ARVC")

    model_file = path.join(args.experiment_directory, str(args.checkpoint) + '.model')
    output_dirs = make_dirs(args.output_directory, use_mc)
    # we create a trainer
    n_classes = len(dta_settings.tissue_structure_labels)
    n_channels_input = 1
    transfer_learn = False
    # IMPORTANT!!! if necessary enable transfer learning settings
    if experiment_settings.dataset != dta_settings.dataset:
        n_classes = len(get_config(experiment_settings.dataset).tissue_structure_labels)
        transfer_learn = True
        print("INFO - transfer learning: trained on nclasses {}".format(n_classes))

    trainer, pad = get_trainer(experiment_settings, n_classes, n_channels_input, model_file=model_file)
    trainer.evaluate_with_dropout = use_mc
    test_results = ARVCTestResult()
    # if patid is not None e.g. "NL256100_1" then we do a single evaluation
    testset_generator = get_test_set_generator(args, experiment_settings, dta_settings, patid=args.patid, all_frames=args.all_frames)
    # we're evaluating patient by patient. one patient can max have 4 volumes if all tissue structures are
    # annotated in separate phases.
    pat_saved, c_phase_saved, sample_saved, phase_results, result_obj = None, None, None, None, None
    for sample in testset_generator:
        image, reference = sample['image'], sample['reference']
        pat_id, phase_id, frame_id = sample['patient_id'], sample['cardiac_phase'], sample['frame_id']
        spacing, original_spacing, direction = sample['spacing'], sample['original_spacing'], sample['direction']

        if pat_saved != pat_id:
            check_save(pat_saved, sample_saved, result_obj, args, experiment_settings, output_dirs, type_of_map)
        result_obj = prepare_result_obj(pat_saved, sample, result_obj, experiment_settings)
        # get ignore_labels (numpy array of shape [n_classes]. Add batch dim in front with None
        ignore_labels, merge_results, phase_results = prepare_evaluation(sample, pat_saved, c_phase_saved,
                                                                         phase_results)
        if transfer_learn:
            reference, ignore_labels = prepare_transfer_learning(n_classes, reference, dta_settings.cls_translate,
                                                                 ignore_labels)
        if pad > 0:
            # image has shape [z, y, x] so pad last two dimensions
            image = np.pad(image, ((0,0), (pad, pad), (pad, pad)), mode="edge") # "'constant', constant_values=(0,))

        image = image[:, None]  # add extra dim of size 1
        image = torch.from_numpy(image)
        pat_predictions = Predictions()
        with torch.set_grad_enabled(False):
            for s in range(args.samples):
                output = trainer.predict(image)
                if ignore_labels is not None:
                    # ignore_labels if not None is a binary vector of size n_classes (target dataset)
                    pred_mask = get_loss_mask(output['softmax'], ignore_labels[None])
                    soft_probs = output['softmax'].detach().numpy() * pred_mask.detach().numpy()
                else:
                    soft_probs = output['softmax'].detach().numpy()
                aleatoric = None if 'aleatoric' not in output.keys() else np.squeeze(output['aleatoric'])
                pat_predictions(soft_probs, cardiac_phase_tag=phase_id, pred_logits=None)
                # torch.cuda.empty_cache()

        pred_probs, uncertainties = pat_predictions.get_predictions(compute_uncertainty=True, mc_dropout=use_mc)
        segmentation = np.argmax(pred_probs, axis=1)

        eval_obj = VolumeEvaluation(pat_id, segmentation, reference,
                                    voxel_spacing=spacing, num_of_classes=n_classes,
                                    mc_dropout=use_mc, cardiac_phase=phase_id, ignore_labels=ignore_labels)
        if args.all_frames:
            eval_obj.post_processing_only()
        else:
            eval_obj.fast_evaluate(compute_hd=True)
            phase_results = process_volume_results(test_results, pat_id, phase_id, merge_results,
                                                   phase_results)
        # IMPORTANT: in fast_evaluate we post process the predictions only keeping the largest connected components
        segmentation = eval_obj.pred_labels

        result_obj = process_volume(result_obj, sample, segmentation, uncertainties, aleatoric=aleatoric)
        if transfer_learn and not args.all_frames:
            print("{}: RV/LV {:.2f} {:.2f}".format(eval_obj.patient_id, eval_obj.dice[1], eval_obj.dice[3]))
        del output
        # save patient and phase we just processed, we need this in order to know whether or not to merge the results
        pat_saved, c_phase_saved, sample_saved = pat_id, phase_id, sample

    if not args.all_frames:
        test_results.show_results(transfer_learning=transfer_learn)
    check_save(pat_saved, sample_saved, result_obj, args, experiment_settings, output_dirs, type_of_map)

    if args.save_results:
        fname = path.join(args.output_directory, "results_f" + str(args.fold) + "_{}".format(len(test_results.pat_ids)) +
                          res_suffix)
        test_results.save(filename=fname)
        print("INFO - performance results saved to {}".format(fname))


if __name__ == '__main__':
    main()
