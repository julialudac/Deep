import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class Create_dataset(Dataset):
    """A dataset created from a list of PIL.Images. One minibatch of this
    dataset is a tensor whose first dimension is its number of elements.
    A (complete) minibatch can be divided into 2 tensors:
    - a tensor of inputs, so a nb_elements * 3 * 36 * 36 tensor for a RGB image,
    36*36 the size of the image).
    - a tensor of labels, so a nb_elements tensor.
    """

    def __convert_all_to_tensors__(self):
        """Convert the list of images into a pure tensor,
        with its elements all transforms(normalized)"""
        l = []
        for im in self.data:
            l.append(self.transform(im))
        self.data = torch.stack(l)

    def __init__(self, ims):
        self.data = ims
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])

        # But ims is a list => minibatches will be list. We don't want that,
        # we want minibatches to be tensors themselves.
        self.__convert_all_to_tensors__()

    def __getitem__(self, index):
        """Important: defines one element (= the input as a tensor + the label (a simple nb))
        of a minibatch. We can for example put the labels=0, no matter: We don't use the label.
        """
        return self.data[index], 0  # I want to return a tensor (the translated img) and a label

    def __len__(self):
        return len(self.data)

