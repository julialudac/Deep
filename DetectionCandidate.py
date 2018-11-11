import FramesAtGivenScaledImage as fat


class DetectionCandidate:
    """It represents a square that represents a detection in the image rescaled to
    normal size. Let we call it a subdetection"""

    def __init__(self, score, position, dims):
        """
        :param score: Score associated to the image frame the subdetection comes from
        :param position: Position associated to the image frame the subdetection comes from, BUT in the rescaled image
        :param dims: Dimension tuple (wifth, height) of the image frame the subdetection comes from, BUT in the rescaled image
        Dimensions of each detection (in the resized image). For example, if the sliding window
        was sliding an image that has been reduced by /1.2, the dimension of each subdetection will
        be (1.2*36, 1.2*36)"
        """
        self.score = score
        self.position = position
        self.dims = dims

    def get_center(self):
        return (self.position[0] + self.dims[0] / 2, self.position[1] + self.dims[1] / 2)

    def __str__(self):
        """Cool for testing/checking :)"""
        return "score:" + str(self.score) + "; position:" + str(self.position) + "; dims:" + str(self.dims)


def capture_good_positions(framesAtGivenScaledImages):
    """
    :param framesAtGivenScaledImages: From this list, we build a list of DetectionCandidates. These are derived from
    the chunks whose associated score is =>0.
    :return: the list of DetectionCandidates
    """
    detectionCandidates = []
    for fagsi in framesAtGivenScaledImages:
        for i in range(len(fagsi.scores)):
            if fagsi.scores[i] >= 0:
                newdims = (int(36 * fagsi.scaling_factor), int(36 * fagsi.scaling_factor))
                """To get the position in the rescaled image (recall that it is top-left), 
                we first scale linearly the center, and then from the center, we deduced the new position
                """
                old_center = fat.get_center(
                    fagsi.positions[i])  # center in the scaled image (so before rescaling to original size)
                newx = old_center[0] * fagsi.scaling_factor
                newx = int(newx - newdims[0] / 2)
                newy = old_center[1] * fagsi.scaling_factor
                newy = int(newy - newdims[1] / 2)
                detectionCandidates.append(DetectionCandidate(fagsi.scores[i], (newx, newy), newdims))

    return detectionCandidates
