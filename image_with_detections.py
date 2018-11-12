from PIL import Image, ImageDraw, ImageFont
import random


def random_color():
    return random.randint(0,255), random.randint(0,255), random.randint(0,255)


class Image_with_detections:

    def __init__(self, image, subdetections):
        self.image = image
        self.subdetections = subdetections

    def save(self, filename="saved_detections.JPG"):
        """Save the image as well as the kept DetectionCanditates."""
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        font_size = 18
        fnt = ImageFont.truetype("impact.ttf", font_size)
        for subdetection in self.subdetections:
            color = random_color()
            draw.rectangle((subdetection.position[0], subdetection.position[1],
                            subdetection.position[0] + subdetection.dimensions[0], subdetection.position[1] + subdetection.dimensions[1]),
                           outline=color)  # But we can't specify border width :(
            draw.text((subdetection.position[0], subdetection.position[1] - font_size), str(subdetection.score), fill=color, font=fnt)
        image.save(filename, "JPEG")