from PIL import Image
from configuration import *


class Chunking:
    """A static class to slide an image at different scales
    and return the captured sub-images and their positions
    """

    @staticmethod
    def init(image, window_dimensions=(36, 36)):
        """
        :param image: a PIL image to slide with sliding window
        :param window_dimensions: width and height (px) of sliding window
        :return: None
        """
        Chunking.image = image
        Chunking.window_dimensions = window_dimensions

    @staticmethod
    def __get_image_chunks__(image, strides=(1, 1)):
        """
        :param image: a PIL image to slide with sliding window
        :param strides: (stride_x, stride_y)
        :return: the Images but also the upper left positions of the
        sliding windows. (But with stride 1, we have too many images!!! :O)
        """
        chunks_image = []
        window_positions = []  # one position is a pair (x,y)
        w, h = image.size
        (current_x, current_y) = (0, 0)
        swept_x, swept_y = [False, False]
        # sliding on x-axis while not all is swept on x-axis:
        while not swept_x:
            current_y = 0
            swept_y = False
            # sliding on y-axis while not all is swept on y-axis:
            while not swept_y:
                chunks_image.append(image.crop((current_x, current_y,
                                                current_x + Chunking.window_dimensions[0], current_y + Chunking.window_dimensions[1])))
                window_positions.append((current_x, current_y))
                if current_y + Chunking.window_dimensions[1] == h:
                    swept_y = True
                    if current_x + Chunking.window_dimensions[0] == w:
                        swept_x = True
                current_y += strides[1]
                # But if the next windows overlaps with the end on y-axis,
                # we don't respect the stride we put but take the last window
                if current_y + Chunking.window_dimensions[1] > h:
                    current_y = h - Chunking.window_dimensions[1]
            current_x += strides[0]
            # But if the next windows overlaps with the end on x-axis,
            # we don't respect the stride we put but take the last window
            # print(current_x)
            if current_x + Chunking.window_dimensions[0] > w:
                current_x = w - Chunking.window_dimensions[0]
        return chunks_image, window_positions

    @staticmethod
    def get_chunks_of_image_at_different_scales(strides=(1, 1), nb_shrinkages=nb_shrinkages, division_factor=division_factor):
        """
        :param strides: a pair of strides: stride on axis x and stride on axis y
        :param nb_shrinkages: number of rescalings
        :param division_factor: initial factor we divide by to rescale
        :return: Returns the Image chunks at different scales and chunks' positions.
        We get 2 2d lists:
        - scaled_chunks = first dimension is chunks for
        shrinked image with respectively a factor of division_factor, 2*division_factor, etc
        - scaled_positions = first dimension is chunks positions for
        shrinked image with respectively a factor of division_factor, 2*division_factor, etc
        """
        current_division_factor = division_factor
        current_image = Chunking.image
        w, h = Chunking.image.size
        scaled_chunks = []
        scaled_positions = []
        for i in range(nb_shrinkages + 1):
            chunks, window_positions = Chunking.__get_image_chunks__(current_image, strides)
            scaled_chunks.append(chunks)
            scaled_positions.append(window_positions)
            current_image = current_image.resize((int(w / current_division_factor), int(h / current_division_factor)))
            current_image.save("reduced" + str(i) + ".JPG", "JPEG")
            current_division_factor *= 2
            current_w, current_h = current_image.size
            # Of course we stop when the sliding window size is bigger than the image
            if current_w < Chunking.window_dimensions[0] or current_h < Chunking.window_dimensions[1]:
                break
        return scaled_chunks, scaled_positions

