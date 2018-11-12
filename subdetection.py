class Subdetection:
    """It represents a square that represents a detection in the image rescaled to
    normal size. Let we call it a subdetection"""

    def __init__(self, score, position, dimensions):
        """
        :param score: Score associated to the image frame the subdetection comes from
        :param position: Position associated to the image frame the subdetection comes from, BUT in the rescaled image
        :param dimensions: Dimension tuple (wifth, height) of the image frame the subdetection comes from, BUT in the rescaled image
        Dimensions of each detection (in the resized image). For example, if the sliding window
        was sliding an image that has been reduced by /1.2, the dimension of each subdetection will
        be (1.2*36, 1.2*36)"
        """
        self.score = score
        self.position = position
        self.dimensions = dimensions

    def get_center(self):
        return (self.position[0] + self.dimensions[0] / 2, self.position[1] + self.dimensions[1] / 2)

    def __str__(self):
        return "score:" + str(self.score) + "; position:" + str(self.position) + "; dimensions:" + str(self.dimensions)

