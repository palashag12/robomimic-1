"""
Implementation of TD3-BC.
Based on https://github.com/sfujim/TD3_BC
(Paper - https://arxiv.org/abs/1812.02900).

Note that several parts are exactly the same as the BCQ implementation,
such as @_create_critics, @process_batch_for_training, and
@_train_critic_on_batch. They are replicated here (instead of subclassing
from the BCQ algo class) to be explicit and have implementation details
self-contained in this file.
"""
from collections import OrderedDict
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.loss_utils as LossUtils
import sklearn
from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
import numpy as np

maxc = []
minc = []
@register_algo_factory_func("knn")







def algo_config_to_class(algo_config):
    """
    Maps algo config to the TD3_BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    # only one variant of TD3_BC for now
    return KNN, {}
def scale_new_row_to_01(new_row, col_max, col_min):
    """
    Scales a new row vector using the same maximum and minimum values that were used to scale
    the columns of a 2D Numpy array.

    Parameters:
        new_row (ndarray): The new row vector to scale.
        col_max (ndarray): An array containing the maximum value of each column in the original array.
        col_min (ndarray): An array containing the minimum value of each column in the original array.

    Returns:
        ndarray: The scaled new row vector.
    """
    # Scale each element of the new row vector using the corresponding maximum and minimum values
    scaled_new_row = (new_row - col_min) / (col_max - col_min)

    return scaled_new_row
def scale_columns_to_01(array):
    """
    Scales all columns of a 2D Numpy array to between 0 and 1.

    Parameters:
        array (ndarray): The input array to scale.

    Returns:
        tuple: A tuple containing a copy of the input array with all columns scaled to between 0 and 1,
               an array containing the maximum value of each column, and an array containing the minimum
               value of each column.
    """
    # Create a copy of the input array
    scaled_array = np.copy(array)

    # Initialize arrays to store the maximum and minimum values for each column
    col_max = np.zeros(scaled_array.shape[1])
    col_min = np.zeros(scaled_array.shape[1])

    # Iterate over each column in the array
    for col in range(scaled_array.shape[1]):
        # Find the maximum and minimum values in the current column
        col_max[col] = np.max(scaled_array[:, col])
        col_min[col] = np.min(scaled_array[:, col])

        # Scale the values in the current column to between 0 and 1
        scaled_array[:, col] = (scaled_array[:, col] - col_min[col]) / (col_max[col] - col_min[col])

    return scaled_array, col_max, col_min
def knn_regression(X_train, Y_train, X_test, k, beta):
    # initialize an empty list to store the predicted labels
    y_pred = []

    # loop over each test data point
    for i in range(len(X_test)):
        distances = []
        distances_dict = {}
        labels = []
        # calculate the distance between the test data point and each training data point

        for j in range(len(X_train)):


            distance = torch.norm(X_test[i] - X_train[j])
            distances.append(distance)
            distances_dict[distance] = Y_train[j]

        # sort the distances in ascending order

        distances2 = sorted(distances)


        # get the k-nearest neighbors
        neighbors = distances2[:k]

        # calculate the softmax weights for the neighbors
        weights = []
        for neighbor in neighbors:
            weights.append(torch.exp(-beta * neighbor))

        weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weighted_sum = torch.zeros_like(distances_dict[neighbors[0]])
        # calculate the predicted value as the weighted average of the neighbors' labels
        for n in range(len(weights)):
            weighted_sum += weights_tensor[n] * distances_dict[neighbors[n]]

        # add the predicted value to the list of predictions
        y_pred.append(weighted_sum)


    return y_pred

def knn_regression2(KDTree,  Y_train, X_test, k, beta):
    y_pred = []

    # loop over each test data point
    for i in range(len(X_test)):
        global maxc
        global minc


        dist, ind = KDTree.query(scale_new_row_to_01(numpy.asarray(X_test), maxc, minc)[0], k)

        # calculate the distance between the test data point and each training data point



        # sort the distances in ascending order



        # get the k-nearest neighbors


        # calculate the softmax weights for the neighbors
        weights = []
        for neighbor in dist:
            weights.append(torch.exp(-beta * torch.tensor(neighbor)))

        weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weighted_sum = torch.zeros_like(Y_train[ind[0]])
        # calculate the predicted value as the weighted average of the neighbors' labels
        for n in range(len(weights)):
            weighted_sum += weights_tensor[n] * Y_train[ind[n]]

        # add the predicted value to the list of predictions
        y_pred.append(weighted_sum)


    return y_pred


class KNN(PolicyAlgo):
    """
    Default TD3_BC training, based on https://arxiv.org/abs/2106.06860 and
    https://github.com/sfujim/TD3_BC.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.states = []
        self.states_dict = {}
        self.actions = []

    def euclidean_distance(x1, x2):
        return torch.sqrt(torch.sum((x1 - x2) ** 2))






    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """


            #self.states.append(batch['obs']['object'])

        for j in batch['obs']['object']:


            self.states.append(j)

        for i in batch['actions']:
            self.actions.append(i)

        numpy_list = [numpy.asarray(t) for t in self.states]
        scale = scale_columns_to_01(numpy_list)
        global maxc
        global minc
        maxc = scale[1]
        minc = scale[2]

        self.KdTree = KDTree(scale[0])
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """

        return knn_regression2(self.KdTree, self.actions, obs_dict['object'], self.algo_config.k, self.algo_config.beta )


