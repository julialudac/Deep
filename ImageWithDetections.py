from PIL import Image, ImageDraw, ImageFont
import random


def random_color():
    return random.randint(0,255), random.randint(0,255), random.randint(0,255)


class ImageWithDetections:

    def __init__(self, im, subdetections):
        self.im = im
        self.subdetections = subdetections

    def save(self, filename="saved_detections.JPG"):
        """Save the image as well as the kept DetectionCanditates."""
        im = self.im.copy()
        draw = ImageDraw.Draw(im)
        fntsize = 18
        fnt = ImageFont.truetype("impact.ttf", fntsize)
        for subd in self.subdetections:
            randomc = random_color()
            draw.rectangle((subd.position[0], subd.position[1],
                            subd.position[0] + subd.dims[0], subd.position[1] + subd.dims[1]),
                           outline=randomc)  # But we can't specify border width :(
            draw.text((subd.position[0], subd.position[1] - fntsize), str(subd.score), fill=randomc, font=fnt)
        im.save(filename, "JPEG")