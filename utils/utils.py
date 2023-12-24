import argparse
import torch.nn.functional as F
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_numpy(tensor, n_dims=2):
    """Convert a torch tensor to numpy array.
    Args:
        tensor (Tensor): a tensor object to convert.
        n_dims (int): size of numpy array shape
    """
    try:
        nparray = tensor.detach().cpu().clone().numpy()
    except AttributeError:
        raise TypeError('tensor type should be torch.Tensor, not {}'.format(type(tensor)))

    while len(nparray.shape) < n_dims:
        nparray = np.expand_dims(nparray, axis=0)

    return nparray


def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)