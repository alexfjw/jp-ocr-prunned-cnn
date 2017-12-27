from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

def train_test_split_indices(dataset: Dataset):
    """
    Generate a train & test indices from the indices
    """
    indices = np.arange(len(dataset))
    return train_test_split(indices, test_size=0.2, shuffle=True)
