import numpy as np
nb_classes = 6
data = [[2, 3, 4, 0]]

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def one_hot_to_indices(data):
    indices = []
    for el in data:
        indices.append(list(el).index(1))
    return indices


hot = indices_to_one_hot(data, nb_classes)
indices = one_hot_to_indices(hot)
print(hot)
print(data)
print(indices)