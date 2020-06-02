
import argparse
import numpy as np
import os
from os import path
from tqdm import tqdm
from utils.visualizer import Visualizer
import datasets.augmentations
import datasets.ACDC.data
from datasets.ARVC.dataset import ARVCDataset
import yaml
import torch
from utils.trainers import get_trainer
from datasets.data_config import get_config
from torch.utils.data import _utils


if torch.cuda.is_available():
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    # torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


from torch.utils.data.sampler import RandomSampler
from torchvision import transforms


def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args


def saveExperimentSettings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)


def get_network_settings(args):

    if args.network[:3] == "drn" or args.network[:4] == "unet":
        args.lr_schedule_type = 'default'
        args.lr_decay_after = 25000
        args.patch_size = tuple((128, 128))
        if args.augmentations is None:
            args.augmentations = "default"
        args.resample = False
        args.weight_decay = 0.0005

    elif args.network[:4] == "dcnn":
        args.lr_schedule_type = 'cyclic'
        args.lr_decay_after = 10000
        args.patch_size = tuple((151, 151))
        args.augmentations = "default"
        args.resample = True if args.dataset == "ACDC" else False
        args.weight_decay = 0.0001

    return args


def get_train_augmentations(args, rs, pad):
    if args.augmentations == "advanced":
        print("WARNING - using advanced augmentations for training")
        training_augmentations = [datasets.augmentations.RandomPerspective(np.random.RandomState(8210)),
                                  datasets.augmentations.PadInput(pad, args.patch_size),
                                  datasets.augmentations.RandomCrop(args.patch_size, input_padding=pad, rs=rs),
                                  datasets.augmentations.RandomMirroring(-2, rs=rs),
                                  datasets.augmentations.RandomMirroring(-1, rs=rs),
                                  datasets.augmentations.RandomRotation((-2, -1), rs=rs),
                                  datasets.augmentations.RandomIntensity(rs=np.random.RandomState(89219)),
                                  datasets.augmentations.ToTensor()]
    else:
        print("WARNING - using standard augmentations for training")
        training_augmentations = [datasets.augmentations.PadInput(pad, args.patch_size),
                                  datasets.augmentations.RandomCrop(args.patch_size, input_padding=pad, rs=rs),
                                  datasets.augmentations.BlurImage(sigma=0.9),
                                  datasets.augmentations.RandomRotation((-2, -1), rs=rs),
                                  datasets.augmentations.ToTensor()]
    return training_augmentations


def get_datasets(args, dta_settings, train_transforms, val_transforms):
    if args.dataset == 'ACDC':
        training_set = datasets.ACDC.data.ACDCDataset('training',
                                                      fold=args.fold,
                                                      root_dir=dta_settings.short_axis_dir,
                                                      resample=args.resample,
                                                      transform=train_transforms,
                                                      limited_load=args.limited_load,
                                                      resample_zaxis=args.super_resolution)
        validation_set = datasets.ACDC.data.ACDCDataset('validation',
                                                        fold=args.fold,
                                                        root_dir=dta_settings.short_axis_dir,
                                                        resample=args.resample,
                                                        transform=val_transforms,
                                                        resample_zaxis=args.super_resolution)

    elif args.dataset == 'ACDC_full':
        training_set = datasets.ACDC.data.ACDCDataset('full',
                                                      fold=args.fold,
                                                      root_dir=dta_settings.short_axis_dir,
                                                      resample=args.resample,
                                                      transform=train_transforms,
                                                      limited_load=args.limited_load,
                                                      resample_zaxis=args.super_resolution)
        validation_set = datasets.ACDC.data.ACDCDataset('validation',
                                                        fold=args.fold,
                                                        root_dir=dta_settings.short_axis_dir,
                                                        resample=args.resample,
                                                        transform=val_transforms,
                                                        resample_zaxis=args.super_resolution)

    elif args.dataset == 'ARVC':
        training_set = ARVCDataset('training', root_dir=dta_settings.short_axis_dir,
                                   resample=args.resample,
                                   transform=train_transforms,
                                   limited_load=args.limited_load)
        validation_set = ARVCDataset('validation',
                                     root_dir=dta_settings.short_axis_dir,
                                     resample=args.resample,
                                     transform=val_transforms,
                                     limited_load=args.limited_load)

    return training_set, validation_set


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('output_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('-p', '--port', type=int, default=8030)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, choices=['ce', 'dice', 'dicev2', 'dicev3', 'brier', 'brierv2'], default='ce')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--number_of_workers', type=int, default=2)
    parser.add_argument('--store_model_every', type=int, default=10000)
    parser.add_argument('--store_curves_every', type=int, default=100)
    parser.add_argument('--update_visualizer_every', type=int, default=100)
    parser.add_argument('--network', type=str, choices=['dcnn', 'drn', 'unet', 'unet_mc', 'drn_mc', 'dcnn_mc',
                                                        'dcnn_mcc', 'drn_mcc'],
                        default='dcnn')
    parser.add_argument('--dataset', type=str, choices=['ACDC', 'ARVC', 'ACDC_full'], default='ACDC')
    parser.add_argument('--augmentations', type=str, choices=['default', 'advanced'], default='default')
    parser.add_argument('--limited_load', action='store_true')
    parser.add_argument('--drop_prob', type=float, default=0.)
    parser.add_argument('--super_resolution', action='store_true', help="upsampling in through-plane direction")
    args = parser.parse_args()
    if args.network[-2:] == "mc" or args.network[-3:] == "mcc" and args.drop_prob == 0.:
        args.drop_prob = 0.1
        print("WARNING - parse_args - Setting drop-prob - {:.3f}".format(args.drop_prob))

    return args


def main():
    # first we obtain the user arguments, set random seeds, make directories, and store the experiment settings.
    args = parse_args()
    # Set resample always to True for ACDC
    args = get_network_settings(args)
    # End - overwriting args
    args.patch_size = tuple(args.patch_size)
    torch.manual_seed(5431232439)
    torch.cuda.manual_seed(5431232439)
    rs = np.random.RandomState(78346)
    os.makedirs(args.output_directory, exist_ok=True)
    saveExperimentSettings(args, path.join(args.output_directory, 'settings.yaml'))
    print(args)
    dta_settings = get_config(args.dataset)

    # we create a trainer
    n_classes = len(dta_settings.tissue_structure_labels)
    n_channels_input = 1

    trainer, pad = get_trainer(args, n_classes, n_channels_input)

    # we initialize datasets with augmentations.
    training_augmentations = get_train_augmentations(args, rs, pad)
    validation_augmentations = [
        datasets.augmentations.PadInput(pad, args.patch_size),
        datasets.augmentations.RandomCrop(args.patch_size, input_padding=pad, rs=rs),
        datasets.augmentations.BlurImage(sigma=0.9),
        datasets.augmentations.ToTensor()]

    training_set, validation_set = get_datasets(args, dta_settings, transforms.Compose(training_augmentations)
                                                , transforms.Compose(validation_augmentations))

    # now we create dataloaders
    tra_sampler = RandomSampler(training_set, replacement=True, num_samples=args.batch_size * args.max_iters)
    val_sampler = RandomSampler(validation_set, replacement=True, num_samples=args.batch_size * args.max_iters)

    data_loader_training = torch.utils.data.DataLoader(training_set,
                                                       batch_size=args.batch_size,
                                                       sampler=tra_sampler,
                                                       num_workers=args.number_of_workers,
                                                       collate_fn=None)  # _utils.collate.default_collate

    data_loader_validation = torch.utils.data.DataLoader(validation_set,
                                                         batch_size=args.batch_size,
                                                         sampler=val_sampler,
                                                         num_workers=args.number_of_workers,
                                                         collate_fn=None)

    # and finally we initialize something for visualization in visdom
    env_suffix = "f" + str(args.fold) + args.output_directory.split("_")[-1]
    vis = Visualizer('Segmentation{}-{}_{}'.format(args.dataset, args.network, env_suffix), args.port,
                     'Learning curves of fold {}'.format(args.fold),
                     ['training', 'validation', 'aleatoric'])
    #
    try:
        for it, (training_batch, validation_batch) in tqdm(enumerate(zip(data_loader_training, data_loader_validation)),
                                                           desc='Training',
                                                           total=args.max_iters):

            # store model
            if not trainer._train_iter % args.store_model_every:
                trainer.save(args.output_directory)

            # store learning curves
            if not trainer._train_iter % args.store_curves_every:
                trainer.save_losses(args.output_directory)

                # visualize example from validation set
                if not trainer._train_iter % args.update_visualizer_every and trainer._train_iter > 20:
                    image = validation_batch['image'][0][None]
                    val_output = trainer.predict(image)
                    prediction = val_output['predictions']
                    reference = validation_batch['reference'][0]
                    val_patient_id = validation_batch['patient_id'][0]

                    image = image.detach().numpy()
                    prediction = prediction.detach().numpy().astype(float)  # .transpose(1, 2, 0)
                    reference = reference.detach().numpy().astype(float)
                    if pad > 0:
                        # Note: image has shape [batch, 1, x, y], we get rid off extra padding in last two dimensions
                        vis.image((image[0, 0, pad:-pad, pad:-pad] ** .5), 'padded image {}'.format(val_patient_id), 12)
                    else:
                        vis.image((image[0] ** .5), 'image {}'.format(val_patient_id), 11)
                    vis.image(reference / 3, 'reference', 13)
                    vis.image(prediction / 3, 'prediction', 14)  # used log_softmax values
                    if 'aleatoric' in val_output.keys():
                        vis.image(val_output['aleatoric'] / 0.9, 'aleatoric', 15)  #
                    # vis.image((prediction >= 0.5).astype(float), 'binary prediction', 15)
                    # visualize learning curve
                    vis(trainer.current_training_loss, trainer.current_validation_loss,
                        trainer.current_aleatoric_loss)  # plot learning curve

            # train on training mini-batch
            trainer.train(training_batch['image'].to(device),
                          training_batch['reference'].to(device),
                          ignore_label=None if 'ignore_label' not in training_batch.keys() else training_batch[
                              'ignore_label'])
            # evaluate on validation mini-batch
            trainer.evaluate(validation_batch['image'].to(device),
                             validation_batch['reference'].to(device),
                             ignore_label=None if 'ignore_label' not in validation_batch.keys() else validation_batch[
                                'ignore_label'])

    except KeyboardInterrupt:
        print('interrupted')

    finally:
        trainer.save(args.output_directory)
        trainer.save_losses(args.output_directory)


if __name__ == '__main__':
    main()

# CUDA_LAUNCH_BLOCKING=1
# hello
# no_proxy=localhost CUDA_VISIBLE_DEVICES=3 python train.py ~/expers/drn_mc_ce -l=0.001 --loss=ce --network=drn --dataset=ACDC --limited_load
