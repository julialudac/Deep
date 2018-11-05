import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class CustomDatasetFromImages(Dataset):
    """A dataset created from a list of PIL.Images. One minibatch of this
    dataset is a tensor whose first dimension is its number of elements.
    A (complete) minibatch can be divided into 2 tensors:
    - a tensor of inputs, so a nbelements * 1 * 36 * 36 tensor (1 because grayscale,
    36*36 the size of the image).
    - a tensor of labels, so a nbelements tensor.
    """

    def convert_all_to_tensor(self):
        """Convert the list of images into a pure tensor,
        with its elements all transformzs(normalized)"""
        l = []
        for im in self.data:
            tensored = self.transform(im)
            l.append(tensored)
        self.data = torch.stack(l)

    def __init__(self, ims):
        self.data = ims
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        # But ims is a list => minibatches will be list. We don't want that,
        # we want minibatches to be tensors themselves.
        self.convert_all_to_tensor()

    def __getitem__(self, index):
        """Important: defines one element (= the input as a tensor + the label (a simple nb))
        of a minibatch. For now, we can for example put the labels=0, no matter."""
        return (self.data[index], 0)  # I want to return a tensor (the translated img) and a label

    def __len__(self):
        return len(self.data)