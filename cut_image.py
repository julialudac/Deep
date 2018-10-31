from PIL import Image


class Chunking:
    """A static class to slide an image at different scales
    and return the captured sub-images and their positions
    """

    @staticmethod
    def init(image, windims = (36,36)):
        """
        :param image: a PIL image to slide with sliding window
        :param windims: width and heigh (px) of sliding window
        :return:
        """
        Chunking.im = image
        Chunking.windims = windims

    @staticmethod
    def get_imgchunks(im, strides=(1, 1)):
        """
        :param strides: (stridex, stridey)
        :return: the Images but also the upper left positions of the
        sliding windows. (But with stride 1, we have too many images!!! :O)
        """
        chunks = []
        winpositions = []  # one position is a pair (x,y)
        w, h = im.size
        (curx, cury) = (0, 0)
        sweptx, swepty = [False, False]
        # sliding on x-axis while not all is swept on x-axis:
        while not sweptx:
            # print("curx:", curx)
            cury = 0
            swepty = False
            # sliding on y-axis while not all is swept on y-axis:
            while not swepty:
                # print("cury:", cury)
                chunks.append(im.crop((curx, cury,
                                          curx + Chunking.windims[0], cury + Chunking.windims[1])))
                winpositions.append((curx, cury))
                if cury + Chunking.windims[1] == h:
                    swepty = True
                    if curx + Chunking.windims[0] == w:
                        sweptx = True
                cury += strides[1]
                # But if the next windows overlaps with the end on y-axis,
                # we don't respect the stride we put but take the last window
                if cury + Chunking.windims[1] > h:
                    cury = h - Chunking.windims[1]
            curx += strides[0]
            # But if the next windows overlaps with the end on x-axis,
            # we don't respect the stride we put but take the last window
            # print(curx)
            if curx + Chunking.windims[0] > w:
                curx = w - Chunking.windims[0]
        return chunks, winpositions

    @staticmethod
    def get_imgchunks_atdiffscales(strides=(1, 1), nbshrinkages=3, divfactor=1.2):
        """
        :param strides: a pair of strides: stride on axis x and stride on axis y
        :param nbshrinkages: number of rescalings
        :param divfactor: initial factor we divide by to rescale
        :return: Returns the Image chunks at different scales and chunks' positions.
        We get 2 2d lists:
        - scaleschunks = first dimension is chunks for
        shrinked image with respectively a factor of divfactor, 2*divfactor, etc
        - scalespositions = first dimension is chunks positions for
        shrinked image with respectively a factor of divfactor, 2*divfactor, etc
        """
        curdivfactor = divfactor
        curim = Chunking.im
        w, h = Chunking.im.size
        scaleschunks = []
        scalespositions = []
        for i in range(nbshrinkages+1):
            chunks, winpositions = Chunking.get_imgchunks(curim, strides)
            scaleschunks.append(chunks)
            scalespositions.append(winpositions)
            curim = curim.resize((int(w / curdivfactor), int(h / curdivfactor)))
            curim.save("reduced" + str(i) + ".JPG", "JPEG")
            curdivfactor *= 2
            curw, curh = curim.size
            # Of course we stop when the sliding window size is bigger than the image
            if curw < Chunking.windims[0] or curh < Chunking.windims[1]:
                break
        return scaleschunks, scalespositions

