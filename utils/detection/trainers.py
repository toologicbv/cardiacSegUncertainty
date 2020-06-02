import torch
import torch.nn as nn
import numpy as np
from os import path
import torchsummary

from networks.detection.region_detector import RegionDetector
from networks.detection.resnet import DRNDetect, drn_d_22, SimpleRSN, BasicBlock, CombinedRSN
from utils.detection_metrics import compute_eval_metrics
from networks.detection.general_setup import config_detector

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    Tensor = torch.FloatTensor
    device = 'cpu'


def get_trainer(args, architecture, model_file=None):
    # architecture is a dictionary with keys from general setup for detector model
    drop_prob = architecture.get("drop_prob", 0.5)
    n_channels_input = architecture.get('n_channels_input')
    n_classes = architecture.get('n_classes')
    weight_decay = architecture.get('weight_decay')
    fn_penalty_weight = architecture.get("fn_penalty_weight", 0)
    fp_penalty_weight = architecture.get('fp_penalty_weight', 0.)

    if args.network == 'rd1':
        trainer = RDSimpleTrainer(learning_rate=args.learning_rate,
                                  model_file=model_file,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=weight_decay,
                                  n_channels_input=n_channels_input,
                                  n_classes=n_classes,
                                  drop_prob=drop_prob,
                                  fn_penalty_weight=fn_penalty_weight,
                                  fp_penalty_weight=fp_penalty_weight)
    elif args.network == 'drn':
        trainer = DRNTrainer(learning_rate=args.learning_rate,
                                  model_file=model_file,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=weight_decay,
                                  n_channels_input=n_channels_input,
                                  n_classes=n_classes,
                                  drop_prob=drop_prob,
                                  fn_penalty_weight=fn_penalty_weight,
                                  fp_penalty_weight=fp_penalty_weight)
    elif args.network == 'rsn':
        trainer = RSNTrainer(learning_rate=args.learning_rate,
                                  model_file=model_file,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=weight_decay,
                                  n_channels_input=n_channels_input,
                                  n_classes=n_classes,
                                  drop_prob=drop_prob,
                                  fn_penalty_weight=fn_penalty_weight,
                                  fp_penalty_weight=fp_penalty_weight)
    elif args.network == 'rsnup':
        trainer = CombinedRSNTrainer(learning_rate=args.learning_rate,
                                  model_file=model_file,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=weight_decay,
                                  n_channels_input=n_channels_input,
                                  n_classes=n_classes,
                                  drop_prob=drop_prob,
                                  fn_penalty_weight=fn_penalty_weight,
                                  fp_penalty_weight=fp_penalty_weight)
    else:
        raise ValueError("ERROR - network {} is not supported".format(args.network))
    return trainer


class Trainer(object):
    def __init__(self,
                 model,
                 n_classes=2,
                 learning_rate=0.001,
                 decay_after=10000,
                 weight_decay=0.,
                 model_file=None, *args, **kwargs):

        self.model = model
        self.model.cuda()
        self.init_lr = learning_rate
        self.n_classes = n_classes
        self.n_channels_input = kwargs.get('n_channels_input')
        self.use_penatly_scheduler = kwargs.get('use_penalty_scheduler', False)
        self.training_losses = list()
        self.validation_losses = list()
        self.validation_metrics = {'prec': 0, 'rec': 0, 'pr_auc': 0, 'roc_auc': 0, 'total_voxel_count': 0,
                                   'detected_voxel_count': 0, 'tp_slice': 0, 'fp_slice': 0, 'tn_slice': 0 ,
                                   'fn_slice': 0}

        self._train_iter = 0

        self.current_training_loss = 0.
        self.current_validation_loss = 0.
        self.decay_after = decay_after
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True,
                                          weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,
        #                                   weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_after)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.5, milestones=[2000, 5000])
        self.loss_function = nn.NLLLoss()
        self.fn_penalty_weight = kwargs.get("fn_penalty_weight")
        self.fp_penalty_weight = kwargs.get("fp_penalty_weight")
        self.decay_penatly_after = 15000
        self.decay_penatly_step = 5000
        print("Trainer - fn_penalty {:.3f} fp_penalty {:.3f}".format(self.fn_penalty_weight, self.fp_penalty_weight))
        if self.use_penatly_scheduler:
            print("Trainer - use scheduler to decrease fn_penalty during training")
        self.weight_decay = kwargs.get('weight_decay')
        if model_file:
            self.load(model_file)
        if kwargs.get('verbose', False):
            torchsummary.summary(self.model, (self.n_channels_input, 80, 80))

    def train(self, images, ref_labels, y_labels_seg=None):
        self.model.train()
        self.optimizer.zero_grad()
        if self.use_penatly_scheduler:
            self._loss_penalty_scheduler()
        output = self.model(images)
        batch_size, channels, _, _ = output["log_softmax"].size()
        loss = self.get_loss(output["log_softmax"].view(batch_size, channels, -1), ref_labels,
                             pred_probs=output["softmax"])
        loss.backward()
        self._train_iter += 1
        self.optimizer.step()
        self.scheduler.step()
        if self._train_iter == self.decay_after:
            print("WARNING ---- LR ", self.scheduler.get_lr())
        self.current_training_loss = loss.detach().cpu().numpy()
        self.training_losses.append(self.current_training_loss)

    def evaluate(self, val_batch, compute_metrics=True, keep_batch=False, batch_size=100, heat_map_handler=None):
        self.model.eval()
        losses, metrics, self.val_segs = [], {k: list() for k in self.validation_metrics.keys()}, []

        for val_image, val_labels in val_batch(batch_size=batch_size, keep_batch=keep_batch):
            with torch.set_grad_enabled(False):
                val_labels = val_labels[config_detector.max_grid_spacing]
                output = self.model(val_image)
            if "softmax_y" in output.keys():
                self.val_segs.append(output["softmax_y"])
            batch_size, channels, _, _ = output["log_softmax"].size()
            loss = self.get_loss(output["log_softmax"].view(batch_size, channels, -1), val_labels,
                                 pred_probs=output["softmax"])
            if keep_batch:
                val_batch.batch_pred_probs.append(np.squeeze(output["softmax"].detach().cpu().numpy()))
                metrics['total_voxel_count'].append(np.sum(val_batch.keep_batch_target_counts[-1]))
                pred_region_lbl = np.argmax(val_batch.batch_pred_probs[-1], axis=0).flatten()
                detected_voxels = val_batch.keep_batch_target_counts[-1] * pred_region_lbl
                metrics['detected_voxel_count'].append(np.sum(detected_voxels))
                pred_pos_slice = int(np.sum(pred_region_lbl) > 0)
                metrics['tp_slice'].append(1 if metrics['total_voxel_count'][-1] > 0 else 0)
                metrics['tn_slice'].append(1 if metrics['total_voxel_count'][-1] == 0 else 0)
                metrics['fp_slice'].append(1 if pred_pos_slice == 1 and metrics['tp_slice'][-1] == 0 else 0)
                metrics['fn_slice'].append(1 if pred_pos_slice == 0 and metrics['tp_slice'][-1] == 1 else 0)

                if heat_map_handler is not None:
                    process_results(val_batch, heat_map_handler, val_batch.batch_pred_probs[-1],
                                    val_labels.detach().cpu().numpy())

            losses.append(loss.detach().cpu().numpy())
            if compute_metrics:
                self.compute_perf_metrics(val_labels.detach().cpu().numpy(), output["softmax"].detach().cpu().numpy(),
                                          metrics)
        self.current_validation_loss = np.mean(np.array(losses))
        if compute_metrics:
            if len(np.array(metrics['rec'])) != 0:
                self.validation_metrics['rec'] = np.mean(np.array(metrics['rec']))
            if len(np.array(metrics['prec'])) != 0:
                self.validation_metrics['prec'] = np.mean(np.array(metrics['prec']))
            if len(np.array(metrics['pr_auc'])) != 0:
                self.validation_metrics['pr_auc'] = np.mean(np.array(metrics['pr_auc']))

            self.validation_metrics['detected_voxel_count'] = np.sum(np.array(metrics['detected_voxel_count']))
            self.validation_metrics['total_voxel_count'] = np.sum(np.array(metrics['total_voxel_count']))
            self.validation_metrics['tp_slice'] = np.sum(np.array(metrics['tp_slice']))
            self.validation_metrics['fp_slice'] = np.sum(np.array(metrics['fp_slice']))
            self.validation_metrics['fn_slice'] = np.sum(np.array(metrics['fn_slice']))
            self.validation_metrics['tn_slice'] = np.sum(np.array(metrics['tn_slice']))

    @staticmethod
    def compute_perf_metrics(ref_labels, pred_probs, result_dict):
        f1, roc_auc, pr_auc, prec, rec, fpr, tpr, precision_curve, recall_curve = \
            compute_eval_metrics(ref_labels, np.argmax(pred_probs, axis=1), pred_probs[:, 1])

        if prec != -1:
            result_dict['prec'].append(prec)
            result_dict['rec'].append(rec)
            result_dict['pr_auc'].append(pr_auc)
            result_dict['roc_auc'].append(roc_auc)

    def get_loss(self, log_pred_probs, lbls, pred_probs=None):
        """

        :param log_pred_probs: LOG predicted probabilities [batch_size, 2, w * h]
        :param lbls: ground truth labels [batch_size, w * h]
        :param pred_probs: [batch_size, 2, w * h]
        :return: torch scalar
        """
        # print("INFO - get_loss - log_pred_probs.shape, lbls.shape ", log_pred_probs.shape, lbls.shape)
        # NOTE: this was a tryout (not working) for hard negative mining
        # batch_loss_indices = RegionDetector.hard_negative_mining(pred_probs, lbls)
        # b_loss_idx_preds = batch_loss_indices.unsqueeze(1).expand_as(log_pred_probs)
        # The input given through a forward call is expected to contain log-probabilities of each class
        b_loss = self.loss_function(log_pred_probs, lbls)

        # pred_probs last 2 dimensions need to be merged because lbls has shape [batch_size, w, h ]
        pred_probs = pred_probs.view(pred_probs.size(0), 2, -1)
        fn_soft = pred_probs[:, 0] * lbls.float()
        # fn_nonzero = torch.nonzero(fn_soft.data).size(0)
        batch_size = pred_probs.size(0)
        fn_soft = torch.sum(fn_soft) * 1 / float(batch_size)
        # same for false positive
        ones = torch.ones(lbls.size()).cuda()
        fp_soft = (ones - lbls.float()) * pred_probs[:, 1]
        # fp_nonzero = torch.nonzero(fp_soft).size(0)
        fp_soft = torch.sum(fp_soft) * 1 / float(batch_size)
        # print(b_loss.item(), (self.fn_penalty_weight * fn_soft + self.fp_penalty_weight * fp_soft).item())
        b_loss = b_loss + self.fn_penalty_weight * fn_soft + self.fp_penalty_weight * fp_soft

        return b_loss

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

    def _loss_penalty_scheduler(self):
        if self._train_iter % self.decay_penatly_step == 0 and self._train_iter > self.decay_penatly_after:
            former_penalty = self.fn_penalty_weight
            self.fn_penalty_weight -= 0.05
            print("WARNING - lowering fn_penalty_weight from {:.3f} to {:.3f}".format(former_penalty,
                                                                                      self.fn_penalty_weight))

    def save_losses(self, output_path):
        np.savez(path.join(output_path, 'losses.npz'), validation=self.validation_losses, training=self.training_losses)


def process_results(eval_batch, heat_map_handler, pred_probs, ref_labels):
    """

    :param eval_batch: BatchHandler object
    :param heat_map_handler: HeatMapHandler object
    :param pred_probs: numpy array [2, w, h] (grid size)
    :param ref_labels: numpy array [w, h]
    :return: n.a.
    """
    # batch_dta_slice_ids is a list of test slice ids, starting from 0...
    slice_id = eval_batch.batch_dta_slice_ids[-1]
    # test_patient_slice_id is a list of 4-tuples where index 0 contains patient_id
    # the slice id in test_patient_slice_id refers to the #slices for this particular patient
    # IMPORTANT: slice_id = overall slice ids used during testing (counter)
    #            whereas pat_slice_id refers to slices in volume
    #            NOTE: pat_slice_id starts with 1...num_of_slices
    # IMPORTANT: if we process separate cardiac phases, cardiac_phase is actually a frame_id
    #            for combined-datasets this is equal to "ES" and "ED"
    patient_id, pat_slice_id, cardiac_phase, frame_id = eval_batch.data_set.test_patient_slice_id[slice_id]
    num_of_slices = eval_batch.data_set.test_patient_num_of_slices[patient_id]
    # softmax probs will be flatted (from w x h grid to long vector) in add_probs if slice_id is filled
    eval_batch.add_probs(pred_probs[1], slice_id=slice_id)
    contains_ref_labels = 1 if np.count_nonzero(ref_labels) != 0 else 0
    eval_batch.add_gt_labels_slice(contains_ref_labels, slice_id=slice_id)
    org_img_size = eval_batch.batch_org_image_size[-1]
    patch_img_size = eval_batch.batch_patch_size[-1]
    patch_slice_xy = eval_batch.batch_slice_areas[-1]
    np_target_voxel_counts = eval_batch.keep_batch_target_counts[-1]
    heat_map_handler.add_patient_slice_pred_probs(patient_id, pat_slice_id, pred_probs[1], cardiac_phase,
                                                       grid_spacing=eval_batch.max_grid_spacing,
                                                       patch_img_size=patch_img_size, patch_slice_xy=patch_slice_xy,
                                                       org_img_size=org_img_size, num_of_slices=num_of_slices,
                                                       create_heat_map=True)
    eval_batch.add_patient_slice_results(patient_id, pat_slice_id, cardiac_phase, pred_probs[1],
                                         ref_labels, np_target_voxel_counts)


class RDSimpleTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        model = RegionDetector(n_classes=kwargs.get('n_classes'), n_channels=kwargs.get('n_channels_input'),
                               drop_prob=kwargs.get("drop_prob"))
        super().__init__(model, *args, **kwargs)


class DRNTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        n_channels = kwargs.get('n_channels_input')
        n_classes = kwargs.get('n_classes')
        model = DRNDetect(drn_d_22(input_channels=n_channels, num_classes=n_classes, out_map=False,
                                     mc_dropout=True), n_classes)
        super().__init__(model, *args, **kwargs)


class RSNTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        n_channels = kwargs.get('n_channels_input')
        n_classes = kwargs.get('n_classes')
        model = SimpleRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes,
                          drop_prob=kwargs.get("drop_prob"))
        super().__init__(model, *args, **kwargs)


class CombinedRSNTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        n_channels = kwargs.get('n_channels_input')
        n_classes = kwargs.get('n_classes')
        model = CombinedRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes,
                            drop_prob=0.5)
        super().__init__(model, *args, **kwargs)

    def train(self, images, ref_labels, y_labels_seg):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(images)
        batch_size, channels, _, _ = output["log_softmax"].size()
        loss = self.get_loss(output["log_softmax"].view(batch_size, channels, -1), ref_labels,
                             pred_probs=output["softmax"])
        y_labels_seg = torch.LongTensor(torch.from_numpy(y_labels_seg).long()).cuda()
        seg_loss = self.loss_function(output["log_softmax_y"].view(batch_size, channels, -1),
                                      y_labels_seg.view(batch_size, -1))
        loss = loss + 5 * seg_loss
        loss.backward()
        self._train_iter += 1
        self.optimizer.step()
        self.scheduler.step()
        self.current_training_loss = loss.detach().cpu().numpy()
        self.training_losses.append(self.current_training_loss)


if __name__ == '__main__':

    # trainer = RDSimpleTrainer(learning_rate=0.001, model_file=None, decay_after=10000, weight_decay=0.0001,
    # n_classes=2, n_channels_input=3)
    #
    # trainer = DRNTrainer(learning_rate=0.001, model_file=None, decay_after=10000, weight_decay=0.0001, n_classes=2,
    #                      n_channels_input=3, fp_penalty_weight=1, fn_penalty_weight=8)
    trainer = CombinedRSNTrainer(learning_rate=0.001, model_file=None, decay_after=10000, weight_decay=0.0001, n_classes=2,
                                 n_channels_input=3, fp_penalty_weight=1, fn_penalty_weight=8)
    dummy_x = torch.randn(1, 3, 72, 72, device=device)
    dummy_y = torch.randint(0, 1, size=(1, 81), device=device)
    dummy_seg_labels = torch.randint(0, 1, size=(1, 72 * 72), device=device)
    trainer.train(dummy_x, dummy_y, y_labels_seg=dummy_seg_labels.detach().cpu().numpy())
    print(trainer.current_training_loss)
