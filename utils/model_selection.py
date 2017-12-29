from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np


def stratified_test_split(dataset: Dataset, test_size: float, val_size=0, random_state=1):
    """
    Generate a train & test indices from a dataset
    Optionally create a validation indices from the train indices,
    Should be % of train set
    """
    indices = np.arange(len(dataset))
    labels = [label for _, label in dataset]
    x_train, x_test, y_train, y_test = train_test_split(indices,
                                                        labels,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=labels,
                                                        random_state=random_state
                                                        )

    if val_size > 0:
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=val_size,
                                                          shuffle=True,
                                                          stratify=y_train,
                                                          random_state=random_state)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_test, y_train, y_test



def train_test_split_indices(dataset: Dataset, test_size:float):
    """
    Generate a train & test indices from the indices
    """
    indices = np.arange(len(dataset))
    return train_test_split(indices, test_size=test_size, shuffle=True)
