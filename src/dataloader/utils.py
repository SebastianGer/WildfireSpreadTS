import torch
import numpy as np


def get_means_and_stds():
    """
    Returns mean and std values as tensor, computed on unaugmented and unnormalized data.
    We don't clip values, because min/max did not diverge much from the 0.1 and 99.9 percentiles.
    Some variables are not normalized, indicated by mean=0, std=1. These are specifically:
    All variables indicating a direction (wind direction, aspect, forecast wind direction),
    and the categorical land cover type. The active fire feature contains either 0, if no fire is present,
    or the hour of the last active fire detection for this pixel. We only divide these values by 24, to 
    have resulting values between 0 and 1. Keeping no-fire values at 0 should make it easier for the model
    to first learn to predict that the fire is going to burn in the same place tomorrow as it does today, 
    which is a good baseline to start from. 
    Edit: Active fire is now also normalized with non-nan mean and std values. 

    :return: mean and std values per variable, computed on unaugmented and unnormalized data of the training set
    """

    # Compute means and stds as tensor
    means = torch.tensor([ 1.9911e+03,  3.1740e+03,  1.9774e+03,  4.5159e+03,  2.4388e+03,
         9.0682e-01,  3.8273e+00,  0.0000e+00,  2.8300e+02,  2.9927e+02,
         6.2968e+01,  6.2108e-03,  5.8389e+00,  0.0000e+00,  1.2077e+03,
        -1.5352e+00,  0.0000e+00,  3.1297e+01,  1.7633e+00,  0.0000e+00,
         1.8925e+01,  6.5871e-03,  1.8991e-02])
    stds = torch.tensor([1.2092e+03, 1.8830e+03, 2.1950e+03, 2.1733e+03, 1.2353e+03, 4.4515e+00,
        1.6786e+00, 1.0000e+00, 7.1452e+00, 7.7938e+00, 2.3382e+01, 3.5115e-03,
        6.8896e+00, 1.0000e+00, 8.6517e+02, 2.1879e+00, 1.0000e+00, 6.9057e+01,
        1.1619e+00, 1.0000e+00, 6.2801e+00, 3.1985e-03, 5.6511e-01])
    return means, stds


def get_indices_of_degree_features():
    """
    :return: Indices of features that take values in [0,360] and thus will be transformed via sin

    """
    return [7, 13, 19]