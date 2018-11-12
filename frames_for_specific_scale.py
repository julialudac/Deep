
class Frames_for_specific_scale():
    """Represents the images chunks/frames for the scaled image at one size.
    Instances are made before the NN feeding to store the datasets and the positions of the frames/image chunks, 
    and also used after the NN feeding to store the scores"""

    def __init__(self, scaling_factor, dataset, positions, scores):
        """
        :param scaling_factor: The factor used to scale (reduce) the image
        :param dataset: Dataset of image frames => this is a Create_dataset instance
        :param positions: Position tuples (x,y) associated to the frames. Positions are from top left.
        :param scores: A list of scores for each frame, for each class
        """
        self.scaling_factor = scaling_factor
        self.dataset = dataset
        self.positions = positions
        self.scores = []
        for score in scores:
            self.scores.append(score[1]-score[0])
            # Scores associated to the frames. These are scalars from -2 to 2. Filled with NN output, method set_scores

