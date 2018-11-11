
class FramesAtGivenScaledImage():
    """Represents the images chunks/frames for the scaled image at one size.
    Instances are made before the NN feeding to store the datasets and the positions of the frames/image chunks, 
    and also used after the NN feeding to store the scores"""

    def __init__(self, scaling_factor, dataset, positions):
        """
        :param scaling_factor: The factor used to scale (reduce) the image
        :param dataset: Dataset of image frames => this is a CustomDatasetFromImages instance
        :param positions: Position tuples (x,y) associated to the frames. Positions are from top left.
        """
        self.scaling_factor = scaling_factor
        self.dataset = dataset
        self.positions = positions
        self.scores = []  # Scores associated to the frames. These are scalars from -2 to 2. Filled with NN output, method set_scores

    def set_scores(net):
        """
        :param net: Neural network used to compute the scores, given the dataset TODO"""
        pass


def get_center(frame_position):
    """Returns the center associated to frame_position in the scaled image 
    (so the size of a frame is not yet resized:36x36)
    """
    return (frame_position[0] + 18, frame_position[1] + 18)
