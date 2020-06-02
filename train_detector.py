import time
import torch
import numpy as np
import yaml
import os
import argparse
from tqdm import tqdm
from pytz import timezone
from datetime import datetime
from utils.detection.trainers import get_trainer
from datasets.region_detection.create_detector_dataset import create_dataset
from utils.detection.batch_handler import BatchHandler, create_grid_heat_map
from networks.detection.general_setup import config_detector
from utils.visualizer import Visualizer


def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp, Loader=yaml.FullLoader))
    return args


def saveExperimentSettings(args, fname):
    with open(fname, 'w') as fp:
        if isinstance(args, dict):
            yaml.dump(args, fp)
        else:
            yaml.dump(vars(args), fp)


def synthesize_output_dir(args, architecture):
    time_id = str.replace(str.replace(str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                          '%Y-%m-%d %H:%M:%S.%f')[4:-7], ' ', '_'), ":", ""), '-', '')
    mc = "_mc" if args.mc_dropout else ""
    channels = "_{}".format(args.input_channels) if args.input_channels != "allchannels" else ""
    output_directory = "f{}_".format(args.fold) + args.network + mc + channels + "_" + \
                                        str(architecture["fn_penalty_weight"]).replace(".","") + "_" + \
                                        str(architecture["fp_penalty_weight"]).replace(".", "") + \
        "_" + time_id
    return output_directory


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('src_path_data', type=str, help='directory that contains data objects from step 1')
    parser.add_argument('--output_directory', type=str, default=None, help='directory for experiment outputs')
    # parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('--input_channels', type=str, choices=['allchannels', 'umap', 'segmask', 'mronly'],
                        default='umap')
    parser.add_argument('--dt_config_id', type=str, default="fixed_46_31", help="config id error filtering")
    parser.add_argument('-p', '--port', type=int, default=8030)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--mc_dropout', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=20000)
    parser.add_argument('--lr_decay_after', type=int, default=10000)
    parser.add_argument('--store_model_every', type=int, default=10000)
    parser.add_argument('--store_curves_every', type=int, default=100)
    parser.add_argument('--update_visualizer_every', type=int, default=25)
    parser.add_argument('--network', type=str, choices=['rd1', 'drn', 'rsn'], default='rsn')
    parser.add_argument('--dataset', type=str, choices=['ACDC', 'ARVC'], default='ACDC')
    parser.add_argument('--limited_load', action='store_true')
    parser.add_argument('--fn_penalty_weight', type=float, default=None)
    parser.add_argument('--grid_search', action='store_true')

    args = parser.parse_args()
    args.src_path_data = os.path.expanduser(args.src_path_data)
    if args.input_channels == 'umap':
        if args.mc_dropout:
            args.input_channels = 'bmap'
        else:
            args.input_channels = 'emap'
    if args.output_directory is not None:
        args.output_directory = os.path.expanduser(args.output_directory)
    return args


def train(args):
    """

    :param exper_hdl:
    :return:
    """

    torch.manual_seed(5431232439)
    torch.cuda.manual_seed(5431232439)
    torch.backends.cudnn.enabled = True
    np.random.seed(6572345)
    # get fold from segmentation experiment settings. We train detection model per cross-validation fold
    seg_exper_settings = loadExperimentSettings(os.path.join(args.src_path_data, 'settings.yaml'))
    args.fold = seg_exper_settings.fold
    print("WARNING - processing fold {}".format(args.fold))
    config_detector.get_architecture(args.network)
    # set number of input channels for initialization of model
    if args.input_channels != "allchannels":
        if args.input_channels == "mronly":
            config_detector.architecture['n_channels_input'] = 1
        else:
            config_detector.architecture['n_channels_input'] = 2
        print("WARNING - Using {} channels as input".format(config_detector.architecture['n_channels_input']))
    if args.fn_penalty_weight is not None:
        config_detector.architecture["fn_penalty_weight"] = args.fn_penalty_weight
        print("WARNING - Using args fn_penalty_weight {:.3f}".format(config_detector.architecture["fn_penalty_weight"]))
    if args.output_directory is None:
        # synthesize
        output_dir = synthesize_output_dir(args, config_detector.architecture)
        args.output_directory = os.path.join(os.path.join(args.src_path_data, "dt_logs"), output_dir)
    else:
        args.output_directory = os.path.expanduser(args.output_directory)
    os.makedirs(args.output_directory, exist_ok=False)
    saveExperimentSettings(args, os.path.join(args.output_directory, 'settings.yaml'))
    saveExperimentSettings(config_detector.architecture, os.path.join(args.output_directory, 'architecture.yaml'))
    print(args)

    # get dataset
    dataset = create_dataset(args.fold, args.src_path_data, mc_dropout=args.mc_dropout, num_of_input_chnls=3,
                             limited_load=args.limited_load, dt_config_id=args.dt_config_id,
                             cardiac_phases=tuple(('ES', 'ED')))
    # and finally we initialize something for visualization in visdom
    seg_model = args.src_path_data.split("/")[-1]
    dt_log_dir = args.output_directory.split("/")[-1]
    env = 'Detection{}-{}-{}_{}'.format(args.dataset, seg_model.replace("_", '-'), args.input_channels, dt_log_dir)
    vis = Visualizer(env, args.port,
                     'Learning curves of fold {}'.format(args.fold),
                     ['training', 'validation'])
    vis_metrics = Visualizer(env, args.port, 'Grid detection prec/rec metrics fold {}'.format(args.fold),
                     ['precision', 'recall', 'pr_auc'])
    vis_detection_rate = Visualizer(env, args.port, 'Slice/voxel detection rate fold {}'.format(args.fold),
                             ['detection_rate', 'slice_tp_rate', 'slice_tn_rate'])
    do_balance_batch = True
    trainer = get_trainer(args, config_detector.architecture, model_file=None)
    try:
        for _ in tqdm(range(args.max_iters), desc="Train {}".format(args.network)):
            # store model
            if not trainer._train_iter % args.store_model_every:
                trainer.save(args.output_directory)

            # store learning curves
            if not trainer._train_iter % args.store_curves_every:
                trainer.save_losses(args.output_directory)

            # visualize example from validation set
            if not trainer._train_iter % args.update_visualizer_every and trainer._train_iter > 0:
                vis(trainer.current_training_loss, trainer.current_validation_loss)  # plot learning curve

            train_batch = BatchHandler(data_set=dataset, is_train=True, verbose=False,
                                       keep_bounding_boxes=False, input_channels=args.input_channels,
                                       num_of_max_pool_layers=config_detector.architecture['num_of_max_pool'],
                                       app_config=config_detector)
            x_input, ref_labels = train_batch(batch_size=args.batch_size, do_balance=do_balance_batch)
            y_labels = ref_labels[config_detector.max_grid_spacing]
            trainer.train(x_input, y_labels, y_labels_seg=train_batch.batch_labels_per_voxel)
            if not trainer._train_iter % args.update_visualizer_every and trainer._train_iter > 0:
                val_batch = BatchHandler(data_set=dataset, is_train=False, verbose=False,
                                         keep_bounding_boxes=False, input_channels=args.input_channels,
                                         num_of_max_pool_layers=config_detector.architecture['num_of_max_pool'],
                                         app_config=config_detector)
                val_set_size = dataset.get_size(is_train=False)
                val_batch.last_test_list_idx = np.random.randint(0, val_set_size - 101, size=1)
                trainer.evaluate(val_batch, keep_batch=True)
                vis_metrics(trainer.validation_metrics['prec'], trainer.validation_metrics['rec'],
                            trainer.validation_metrics['pr_auc'])
                dt_rate = trainer.validation_metrics['detected_voxel_count'] / trainer.validation_metrics['total_voxel_count']
                dt_slice_tp = trainer.validation_metrics['tp_slice'] / (trainer.validation_metrics['tp_slice'] + \
                              trainer.validation_metrics['fn_slice'])
                dt_slice_tn = trainer.validation_metrics['tn_slice'] / (trainer.validation_metrics['tn_slice'] + \
                              trainer.validation_metrics['fp_slice'])
                vis_detection_rate(dt_rate, dt_slice_tp, dt_slice_tn)
                idx = 12
                patid = val_batch.batch_patient_slice_id[idx][0]
                val_img = val_batch.keep_batch_images[idx][0][0]
                w, h, = val_img.shape
                vis.image((val_img ** .5), 'image {}'.format(patid), 11)
                vis.image((val_batch.keep_batch_images[idx][0][1] / 0.9), 'uncertainty {}'.format(patid), 12)
                vis.image(val_batch.keep_batch_label_slices[idx] / 1.001, 'reference', 13)
                vis.image((val_batch.keep_batch_images[idx][0][2] / 1.001), 'seg mask', 16)
                p = np.squeeze(val_batch.batch_pred_probs[idx])[1]
                heat_map, grid_map, target_lbl_grid = create_grid_heat_map(p, config_detector.max_grid_spacing, w, h,
                                                                           prob_threshold=0.5)
                vis.image((heat_map ** .5), 'grid predictions', 14)
                if args.network == "rsnup":
                    p_mask = np.argmax(np.squeeze(trainer.val_segs[idx]), axis=0)
                    vis.image(p_mask / 1.001, 'predictions', 15)

                del val_batch

    except KeyboardInterrupt:
        print('interrupted')

    finally:
        trainer.save(args.output_directory)
        trainer.save_losses(args.output_directory)


def main():
    args = parse_args()
    if args.grid_search:
        if args.mc_dropout:
            fn_penalty_grid = [0.65, 0.75, 0.85]
        else:
            fn_penalty_grid = [0.7, 0.8, 0.9, 1., ]
        for fn_penalty in fn_penalty_grid:
            args.fn_penalty_weight = fn_penalty
            train(args)
    else:
        train(args)


if __name__ == '__main__':
    main()

"""
no_proxy=localhost CUDA_VISIBLE_DEVICES=3 python train_detector.py ~/expers/acdc/f0/drn_mc_ce -l=0.00001 --network=rsn --batch_size=32 
 --lr_decay_after=10000 --update_visualizer_every=25 --dataset=ACDC --max_iters=20000

"""
