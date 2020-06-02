def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log() 


def build_categorical_aleatoric_loss(samples):
    """
    Build a categorical aleatoric loss function.
    Args:
        sample: the number of samples (T) to perform a Monte Carlo integration
    Returns:
        a callable Keras loss function
    """
    def categorical_aleatoric_loss(y_true, y_pred):
        """
        Return the categorical aleatoric loss for true values and predictions.
        Args:
            y_true: the ground truth values as a onehot Tensor
            y_pred: the predicted values as a onehot/probability Tensor
        Returns:
            categorical aleatoric loss
        """
        # unwrap the logits and sigma from the networks prediction stack
        logits = y_pred[..., 0]
        sigma = y_pred[..., 1]
        # create a list to store the output of simulations in
        simulations = [None] * samples
        # perform the Monte Carlo simulation over the number of samples
        for sample in range(samples):
            # initialize a Gaussian random variable to sample from logit space
            epsilon_t = K.random_normal(K.shape(sigma))
            # sample the logits through the Softmax function
            x = activations.softmax(logits + sigma * epsilon_t)
            # mask logits sample using ground truth and extract using max
            x_c = K.max(x * y_true, axis=-1)
            # subtract the log-sum-exp of the sample from the observed label's
            # logit value to produce this simulations output
            simulations[sample] = x_c - K.logsumexp(x, axis=-1)
        # concatenate the simulations into a tensor with shape (..., samples)
        simulations = K.concatenate(simulations)
        # finish out the loss with the log-mean-exp over the simulations. Use
        # the quotient rule of logarithms to accomplish the mean calculation
        # through the numerically stable logsumexp method, opposed to
        # something like `K.log(K.sum(K.exp(...)) / T)`
        loss = K.logsumexp(simulations, axis=-1) - K.log(float(samples))
        # return the sum over the loss for each pixel i
        return K.sum(loss)

    return categorical_aleatoric_loss