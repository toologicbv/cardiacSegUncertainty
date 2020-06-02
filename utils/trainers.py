import torch
import torch.nn as nn
import numpy as np
from os import path
from torch.distributions import Independent, Normal

if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)

import torchsummary
from utils import losses

from networks import DilatedCNN2D, BayesDilatedCNN2D, DRNSeg, BayesDRNSeg, drn_d_22, FCDenseNet57, DKFZUNet, BayesUNet
from networks import CombinedBayesDilatedCNN2D
from networks import CombinedBayesDRNSeg
from networks import CycleLR


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Cat(nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *input):
        cat = torch.cat(input, self.dim)
        return cat


def get_trainer(args, n_classes, n_channels_input, model_file=None):
    if "drop_prob" not in vars(args).keys():
        args.drop_prob = 0.
        if args.network[-2:] == "mc" or args.network[-3:] == "mcc":
            args.drop_prob = 0.1
            print("WARNING - Setting drop-prob - get - trainer ", args.drop_prob)

    if args.network == 'dcnn':
        trainer = DCNN2D(learning_rate=args.learning_rate,
                         model_file=model_file,
                         decay_after=args.lr_decay_after,
                         lr_schedule_type=args.lr_schedule_type,
                         weight_decay=args.weight_decay,
                         n_classes=n_classes,
                         n_channels=n_channels_input,
                         loss=args.loss)
        pad = 65
    if args.network == 'dcnn_mc':
        trainer = BayesDCNN2D(learning_rate=args.learning_rate,
                              model_file=model_file,
                              lr_schedule_type=args.lr_schedule_type,
                              decay_after=args.lr_decay_after,
                              weight_decay=args.weight_decay,
                              n_classes=n_classes,
                              n_channels=n_channels_input,
                              loss=args.loss)
        pad = 65

    elif args.network == "dcnn_mcc":

        trainer = CombinedBayesDCNN2D(learning_rate=args.learning_rate,
                                      model_file=model_file,
                                      lr_schedule_type=args.lr_schedule_type,
                                      decay_after=args.lr_decay_after,
                                      weight_decay=args.weight_decay,
                                      n_classes=n_classes,
                                      n_channels=n_channels_input,
                                      loss=args.loss)
        pad = 65

    elif args.network == 'drn':
        trainer = DRN2D(learning_rate=args.learning_rate,
                        model_file=model_file,
                        lr_schedule_type=args.lr_schedule_type,
                        decay_after=args.lr_decay_after,
                        weight_decay=args.weight_decay,
                        n_classes=n_classes,
                        n_channels=n_channels_input,
                        norm_layer="batch_norm",
                        loss=args.loss)
        pad = 0
    elif args.network == 'drn_mc':
        if "norm_layer" not in vars(args).keys():
            args.norm_layer = "batch_norm"
        trainer = BayesDRN2D(learning_rate=args.learning_rate,
                             model_file=model_file,
                             lr_schedule_type=args.lr_schedule_type,
                             decay_after=args.lr_decay_after,
                             weight_decay=args.weight_decay,
                             n_classes=n_classes,
                             n_channels=n_channels_input,
                             norm_layer=args.norm_layer,
                             loss=args.loss,
                             drop_prob=args.drop_prob)
        pad = 0
    elif args.network == 'drn_mcc':
        if "norm_layer" not in vars(args).keys():
            args.norm_layer = "batch_norm"
        trainer = CombinedBayesDRN2D(learning_rate=args.learning_rate,
                                     model_file=model_file,
                                     lr_schedule_type=args.lr_schedule_type,
                                     decay_after=args.lr_decay_after,
                                     weight_decay=args.weight_decay,
                                     n_classes=n_classes,
                                     n_channels=n_channels_input,
                                     norm_layer=args.norm_layer,
                                     loss=args.loss,
                                     drop_prob=args.drop_prob)
        pad = 0
    elif args.network == 'unet':
        trainer = UNet2D(learning_rate=args.learning_rate,
                         model_file=model_file,
                         lr_schedule_type=args.lr_schedule_type,
                         decay_after=args.lr_decay_after,
                         weight_decay=args.weight_decay,
                         n_classes=n_classes,
                         n_channels=n_channels_input,
                         loss=args.loss)
        pad = 0
    elif args.network == 'unet_mc':
        trainer = BayesUNetTrainer(learning_rate=args.learning_rate,
                                   model_file=model_file,
                                   lr_schedule_type=args.lr_schedule_type,
                                   decay_after=args.lr_decay_after,
                                   weight_decay=args.weight_decay,
                                   n_classes=n_classes,
                                   n_channels=n_channels_input,
                                   loss=args.loss,
                                   drop_prob=args.drop_prob)
        pad = 0

    return trainer, pad


class Trainer(object):
    def __init__(self,
                 model,
                 loss,
                 n_classes,
                 learning_rate=0.001,
                 decay_after=1000000,
                 weight_decay=0.,
                 model_file=None, *args, **kwargs):
        self.model = model
        self.model.cuda()
        self.init_lr = learning_rate
        self.n_classes = n_classes
        self.evaluate_with_dropout = kwargs.get("use_mc", False)
        self.lr_schedule_type = kwargs.get("lr_schedule_type", 'default')
        self.loss_name = loss
        print("INFO - Trainer ", model.__class__, loss, n_classes)
        self.loss_mask = None
        if loss == 'ce':
            self.criterion = nn.NLLLoss()
            self.criterion_key = 'log_softmax'
        elif loss == 'dice':
            self.criterion = losses.DiceLoss(n_classes)
            self.criterion_key = 'softmax'
        elif loss == 'dicev2':
            self.criterion = losses.DiceLossv2(n_classes)
            self.criterion_key = 'softmax'
        elif loss == "dicev3":
            self.criterion = losses.soft_dice_loss
            self.criterion_key = 'softmax'
        elif loss == "brier":
            self.criterion = losses.BrierLoss(n_classes)
            self.criterion_key = 'softmax'
        elif loss == 'brierv2':
            self.criterion = losses.brier_score_loss
            self.criterion_key = 'softmax'
        else:
            raise ValueError("ERROR - Trainer - loss function unknown {}".format(loss))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True,
                                          weight_decay=weight_decay)
        if self.lr_schedule_type == "default":
            print("INFO - using default lr schedule!")
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_after)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                              amsgrad=False, weight_decay=weight_decay)
            self.scheduler = CycleLR(self.optimizer, alpha_zero=self.init_lr, cycle_length=10000)

        self.training_losses = list()
        self.validation_losses = list()

        self._train_iter = 0

        self.current_training_loss = 0.
        self.current_validation_loss = 0.
        self.current_aleatoric_loss = 0.
        if model_file:
            self.load(model_file)
        if kwargs.get('verbose', False):
            torchsummary.summary(self.model, (kwargs.get('n_channels'), 256, 256))

    def predict(self, image):
        self.model.eval()
        image = image.cuda()
        output = self.model(image)
        _, res = torch.max(output['softmax'].cpu().detach(), 1)
        return {'predictions': res, 'softmax': output['softmax'].cpu().detach()}

    def train(self, image, reference, ignore_label=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(image)
        loss = self.get_loss(output, reference, ignore_label=ignore_label)
        # print("Loss {:.3f}".format(loss.item()))
        loss.backward()
        self._train_iter += 1
        self.optimizer.step()
        self.current_training_loss = loss.detach().cpu().numpy()
        self.training_losses.append(self.current_training_loss)
        self.scheduler.step()

    def evaluate(self, image, reference, ignore_label=None):
        self.model.eval()
        output = self.model(image)
        loss = self.get_loss(output, reference, ignore_label=ignore_label)
        self.current_validation_loss = loss.detach().cpu().numpy()
        self.validation_losses.append(self.current_validation_loss)

    def get_loss(self, output, target, ignore_label=None):
        if ignore_label is not None:
            loss_mask = losses.get_loss_mask(output['softmax'], ignore_label)
            loss_mask = loss_mask.to(device)
            # print(np.prod(loss_mask.shape), np.count_nonzero(loss_mask.cpu().detach().numpy()))
            # l_all = self.criterion(output[self.criterion_key], target)
            return self.criterion(output[self.criterion_key] * loss_mask, target)
            # print("All {:.3f} Masked {:.3f}".format(l_all, l_masked))
        else:
            return self.criterion(output[self.criterion_key], target)

    def load(self, fname):
        state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict['model_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_dict'])
        self._train_iter = state_dict['iteration']
        with np.load(path.join(path.split(fname)[0], 'losses.npz')) as npz:
            self.validation_losses = list(npz['validation'][:self._train_iter + 1])
            self.training_losses = list(npz['training'][:self._train_iter + 1])

    def save(self, output_dir):
        fname = path.join(output_dir, '{:0d}.model'.format(self._train_iter))
        torch.save({'model_dict': self.model.state_dict(),
                    'optimizer_dict': self.optimizer.state_dict(),
                    'iteration': self._train_iter}, fname)
    
    def save_losses(self, output_path):
        np.savez(path.join(output_path, 'losses.npz'), validation=self.validation_losses, training=self.training_losses)


class BayesTrainer(Trainer):

    def predict(self, image):
        self.model.eval(mc_dropout=self.evaluate_with_dropout)
        image = image.cuda()
        output = self.model(image)
        _, res = torch.max(output['softmax'], 1)
        return {'predictions': res.cpu(), 'softmax': output['softmax'].cpu()}

    def evaluate(self, image, reference, ignore_label=None):
        self.model.eval(mc_dropout=self.evaluate_with_dropout)
        output = self.model(image)
        loss = self.get_loss(output, reference, ignore_label=ignore_label)
        self.current_validation_loss = loss.detach().cpu().numpy()
        self.validation_losses.append(self.current_validation_loss)


class DCNN2D(Trainer):
    def __init__(self, *args, **kwargs):

        model = DilatedCNN2D(n_input=kwargs.get('n_channels'), n_classes=kwargs.get('n_classes'))
        super().__init__(model, *args, **kwargs)


class BayesDCNN2D(BayesTrainer):
    def __init__(self, *args, **kwargs):

        model = BayesDilatedCNN2D(n_input=kwargs.get('n_channels'), n_classes=kwargs.get('n_classes'))
        super().__init__(model, *args, **kwargs)


class CombinedBayesDCNN2D(BayesTrainer):
    def __init__(self, *args, **kwargs):

        model = CombinedBayesDilatedCNN2D(n_input=kwargs.get('n_channels'), n_classes=kwargs.get('n_classes'))
        super().__init__(model, *args, **kwargs)
        self.samples = 5
        self.sigma = 1
        self.softmax = nn.Softmax(dim=1)

    def predict(self, image):
        self.model.eval(mc_dropout=self.evaluate_with_dropout)
        image = image.cuda()
        output = self.model(image)
        _, res = torch.max(output['softmax'], 1)
        output = {'predictions': res.cpu().detach(), 'softmax': output['softmax'].cpu().detach(),
                  'aleatoric': output['sigma'].cpu().detach().numpy()}
        return output

    def get_loss(self, output, target, ignore_label=None):
        loss = super().get_loss(output, target, ignore_label)
        eps = 1e-12
        # construct N(0,1) diagonal covariance of size y (output)
        # construct N(0,1) diagonal covariance of size y (output)
        normal = Independent(Normal(loc=torch.FloatTensor(output['logits'].size()).fill_(0).to(device),
                                    scale=torch.FloatTensor(output['logits'].size()).fill_(self.sigma).to(device)), 1)
        # sum ( softmax (distorted softmax probs)) using predicted voxel variances (scale)
        # we then take the log of these
        sum_distorted_softmax = torch.sum(
            torch.stack([self.softmax(output['logits'] + (output['sigma'] * normal.sample())) for _ in
                         torch.arange(self.samples)]), dim=0)
        # sum_distorted_softmax should have shape [batch, nclasses, x, y]
        one_hot = torch.zeros(output['logits'].shape).scatter_(1, target.unsqueeze(1), 1)
        # mask sum_distorted_softmax in order to obtain only the softmax probs for the gt class and take max
        # of the result, which will just select the prob of the gt class (reduce dim 1=nclasses)
        sum_distorted_softmax, _ = torch.max(sum_distorted_softmax * one_hot, 1)
        # sum_distorted_softmax should now have shape [batch, x, y]
        # finally compute the categorical aleatoric loss. We assume we're using Brier loss, otherwise need to
        # scale the aleatoric loss to be in the neighborhood of e.g. CE (0.0001 *)
        if self.loss_name[:5] == "brier":
            aleatoric_scale = 1.
        else:
            aleatoric_scale = 0.0001
        aleatoric_loss = - aleatoric_scale * torch.mean(
            torch.sum(torch.log(sum_distorted_softmax + eps) - np.log(self.samples), dim=(1, 2)), dim=0)
        output['sigma'] = output['sigma'].cpu().detach().numpy()
        output['logits'] = None
        self.current_aleatoric_loss = aleatoric_loss.detach().cpu().numpy()
        # print("Loss / aleatoric loss {:.3f} / {:.3f}".format(loss.item(), self.current_aleatoric_loss.item()))
        return loss + aleatoric_loss


class DRN2D(Trainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        norm_layer = kwargs.get('norm_layer', 'batch_norm')
        model = DRNSeg(drn_d_22(input_channels=n_channels, num_classes=n_classes, out_map=False, norm_layer=norm_layer)
                       , n_classes)
        super().__init__(model, *args, **kwargs)


class BayesDRN2D(BayesTrainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        drop_prob = kwargs.get('drop_prob')
        norm_layer = kwargs.get('norm_layer', 'batch_norm')
        model = BayesDRNSeg(drn_d_22(input_channels=n_channels, num_classes=n_classes, out_map=False,
                                     drop_prob=drop_prob, norm_layer=norm_layer), n_classes)
        super().__init__(model, *args, **kwargs)


class CombinedBayesDRN2D(BayesTrainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        drop_prob = kwargs.get('drop_prob')
        norm_layer = kwargs.get('norm_layer', 'batch_norm')
        model = CombinedBayesDRNSeg(drn_d_22(input_channels=n_channels, num_classes=n_classes, out_map=False,
                                             drop_prob=drop_prob, norm_layer=norm_layer), n_classes)
        super().__init__(model, *args, **kwargs)
        self.samples = 5
        self.sigma = 1
        self.softmax = nn.Softmax(dim=1)

    def predict(self, image):
        self.model.eval(mc_dropout=self.evaluate_with_dropout)
        image = image.cuda()
        output = self.model(image)
        _, res = torch.max(output['softmax'], 1)
        output = {'predictions': res.cpu().detach(), 'softmax': output['softmax'].cpu().detach(),
                  'aleatoric': output['sigma'].cpu().detach().numpy()}
        return output

    def get_loss(self, output, target, ignore_label=None):
        loss = super().get_loss(output, target, ignore_label)
        eps = 1e-12
        # construct N(0,1) diagonal covariance of size y (output)
        # construct N(0,1) diagonal covariance of size y (output)
        normal = Independent(Normal(loc=torch.FloatTensor(output['logits'].size()).fill_(0).to(device),
                                    scale=torch.FloatTensor(output['logits'].size()).fill_(self.sigma).to(device)), 1)
        # sum ( softmax (distorted softmax probs)) using predicted voxel variances (scale)
        # we then take the log of these
        sum_distorted_softmax = torch.sum(
            torch.stack([self.softmax(output['logits'] + (output['sigma'] * normal.sample())) for _ in
                         torch.arange(self.samples)]), dim=0)
        # sum_distorted_softmax should have shape [batch, nclasses, x, y]
        one_hot = torch.zeros(output['logits'].shape).scatter_(1, target.unsqueeze(1), 1)
        # mask sum_distorted_softmax in order to obtain only the softmax probs for the gt class and take max
        # of the result, which will just select the prob of the gt class (reduce dim 1=nclasses)
        sum_distorted_softmax, _ = torch.max(sum_distorted_softmax * one_hot, 1)
        # sum_distorted_softmax should now have shape [batch, x, y]
        # finally compute the categorical aleatoric loss
        aleatoric_loss = -0.0001 * torch.mean(
            torch.sum(torch.log(sum_distorted_softmax + eps) - np.log(self.samples), dim=(1, 2)), dim=0)
        output['sigma'] = output['sigma'].cpu().detach().numpy()
        output['logits'] = None
        self.current_aleatoric_loss = aleatoric_loss.detach().cpu().numpy()
        return loss + aleatoric_loss


class UNet2D(Trainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        model = DKFZUNet(num_classes=n_classes, in_channels=n_channels)
        super().__init__(model, *args, **kwargs)


class BayesUNetTrainer(BayesTrainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        drop_prob = kwargs.get('drop_prob', 0.1)
        model = BayesUNet(num_classes=n_classes, in_channels=n_channels, drop_prob=drop_prob)
        super().__init__(model, *args, **kwargs)


if __name__ == "__main__":
    # trainer = UNet2D(learning_rate=0.001,
    #                             decay_after=25000,
    #                             weight_decay=0.0005,
    #                             n_classes=4,
    #                             n_channels=1,
    #                             loss='ce',
    #                             drop_prob=0.)
    # trainer.model.train(mode=False, mc_dropout=True)
    trainer = BayesDRN2D(learning_rate=0.001,
                    model_file=None,
                    lr_schedule_type="default",
                    decay_after=25000,
                    weight_decay=0.0005,
                    n_classes=4,
                    n_channels=1,
                    loss="ce",
                         drop_prob=0.2)
    trainer.model.train(mode=False, mc_dropout=True)