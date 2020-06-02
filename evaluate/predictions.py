
import numpy as np
from utils.common import compute_entropy


class Predictions(object):

    def __init__(self):
        self.pred_probs = {}
        self.pred_labels = {}
        self.pred_logits = {}
        self.cardiac_phases = {}
        self.mc_dropout = False

    def __call__(self, pred_probs, cardiac_phase_tag, pred_logits=None):
        """

        :param pred_probs: [N, C, w, h] where N=#slices and C=number of classes, can be 8 (combined) or 4
        :param pred_logits: [N, C, w, h]
        :param cardiac_phase_tag:
        :return:
        """
        if not isinstance(pred_probs, np.ndarray):
            pred_probs = pred_probs.detach().cpu().numpy()
        if pred_logits is not None and not isinstance(pred_logits, np.ndarray):
            pred_logits = pred_logits.detach().cpu().numpy()

        pred_probs = np.expand_dims(pred_probs, axis=0)
        if pred_logits is not None:
            pred_logits = np.expand_dims(pred_logits, axis=0)
        if cardiac_phase_tag not in self.cardiac_phases.keys():
            # print("Test_result: new phase {}".format(cardiac_phase_tag))
            self.cardiac_phases[cardiac_phase_tag] = cardiac_phase_tag
            self.pred_probs[cardiac_phase_tag] = pred_probs
            if pred_logits is not None:
                self.pred_logits[cardiac_phase_tag] = pred_logits
        else:
            self.pred_probs[cardiac_phase_tag] = np.vstack([self.pred_probs[cardiac_phase_tag], pred_probs])
            if pred_logits is not None:
                self.pred_logits[cardiac_phase_tag] = np.vstack([self.pred_logits[cardiac_phase_tag], pred_logits])

    def get_logits(self):
        if len(self.pred_logits) == 2:
            logits = np.concatenate((np.mean(self.pred_logits['ES'], axis=0), np.mean(self.pred_logits['ED'], axis=0)))
            # np_array will have return shape [N, C, w, h] see above. We average over number of samples/predictions
            return logits
        else:
            phase_key = list(self.cardiac_phases.keys())[0]
            logits = self.pred_logits[phase_key]
            if logits.shape[0] > 1:
                mean_logits = np.mean(logits, axis=0)
            else:
                mean_logits = np.squeeze(logits, axis=0)
            return mean_logits

    def get_predictions(self, compute_uncertainty=False, mc_dropout=False, agg_func="max"):
        if len(self.pred_probs) == 2:
            print("WARNING - This needs your attention. We haven't implementing this object yet for 2 simultaneous "
                  "cardiac phases!!!!")
            # Dealing with two cardiac phases at the same time
            # average (in case we have more than 1 prediction per voxel) and make sure each sum to one (probabilities!)
            pred_es = np.mean(self.pred_probs['ES'], axis=0)
            pred_es *= 1. / np.sum(pred_es, axis=1, keepdims=True)
            pred_ed = np.mean(self.pred_probs['ED'], axis=0)
            pred_ed *= 1. / np.sum(pred_ed, axis=1, keepdims=True)
            np_array = np.concatenate((pred_es, pred_ed))
            # np_array will have return shape [N, C, w, h] see above. We average over number of samples/predictions
            return np_array
        else:
            # one cardiac phase
            phase_key = list(self.cardiac_phases.keys())[0]
            preds = self.pred_probs[phase_key]
            if preds.shape[0] > 1:
                # we have more than 1 prediction for this volume [s, z, 4, x, y] where s = #models or mc-samples
                mean_preds = np.mean(preds, axis=0)
                # normalize => make sure they sum to ONE (probabilities!)
                mean_preds *= 1./np.sum(mean_preds, axis=1, keepdims=True)
            else:
                mean_preds = np.squeeze(preds, axis=0)

            if compute_uncertainty:
                # Note: Rescaling uncertainty maps between [0, 1]
                if mc_dropout:
                    # we're dealing with mc samples in preds, so we want to capture the stddev of the samples in order to
                    # use them as uncertainty measure: preds has [#samples, z, 4, x, y]
                    umap = np.std(preds, axis=0)  # we still have the stddev per class (now axis=1) [z, nclass, x, y]
                    # Note: rescaling before we average or taking maximum.
                    # Maximum variance for per class is 0.25, because probabilities are between [0, 1]. Extreme case when we have 10
                    # samples (per class) and 5 have zero probability and 5 have probability of one. We're using stddev = sqrt(var)
                    # hence maximum value is 0.5. Therefore we scale by a factor of 2
                    umap *= 2
                    # make sure do not accidentally get values larger than 1, should be impossible
                    umap[umap > 1] = 1
                    if agg_func == "max":
                        umap = np.max(umap, axis=1)
                    else:
                        umap = np.mean(umap, axis=1)
                else:
                    # mean_preds has shape [#slices, nclasses, x, y], so we sum over classes=>axis=1
                    # rescale between [0, 1]. max entropy for four classes is 2, hence we multiply by 0.5
                    # NOTE: we move rescaling to compute_entropy function!!!
                    umap = compute_entropy(mean_preds, dim=1)  # we sum over dim=1 the classes

                # print("Uncertainty map ", np.min(umap), np.max(umap))
                return mean_preds, umap
            return mean_preds
