from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np


def train_test_split_indices(dataset: Dataset, test_size:float):
    """
    Generate a train & test indices from the indices
    """
    indices = np.arange(len(dataset))
    return train_test_split(indices, test_size=test_size, shuffle=True)
