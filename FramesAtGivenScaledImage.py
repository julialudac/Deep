import numpy as np


class FramesAtGivenScaledImage():
    """Represents the images chunks/frames for the scaled image at one size.
    Instances are made before the NN feeding to store the datasets and the positions of the frames/image chunks, 
    and also used after the NN feeding to store the scores"""

    def __init__(self, scaling_factor, dataset, positions, scores):
        """
        :param scaling_factor: The factor used to scale (reduce) the image
        :param dataset: Dataset of image frames => this is a CustomDatasetFromImages instance
        :param positions: Position tuples (x,y) associated to the frames. Positions are from top left.
        :param scores: A list of scores for each frame, for each class
        """
        self.scaling_factor = scaling_factor
        self.dataset = dataset
        self.positions = positions
        self.scores = []
        # TODO: after converting the 2d tensor into a 2d numpy array, get the result of it for the softmax function. I guess the piece of code is somewhere here.
        for score in scores:
            self.scores.append(score[1]-score[0]) # Scores associated to the frames. These are scalars from -2 to 2. Filled with NN output, method set_scores


def get_center(frame_position):
    """Returns the center associated to frame_position in the scaled image 
    (so the size of a frame is not yet resized:36x36)
    """
    return (frame_position[0] + 18, frame_position[1] + 18)


def softmax(my2darray):
    """
    :param my2darray: a 2d numpy array: each element is a vector of 2 elements, and
    we want to apply the softmax to each of these elements
    :return: a 2d numpy array which is has softmaxed elements of myarray
    """
    numerator = np.exp(my2darray) # OK
    #print("numerator:", numerator)
    # We want a matrix to avoid the column vector being transformed into a row vector
    denominator = np.zeros((1,numerator.shape[0]))
    denominator[0] = np.sum(np.exp(my2darray), axis=1) # values OK
    denominator = np.transpose(denominator)
    #print("denominator:", denominator)
    return numerator/denominator