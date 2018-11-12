from PIL import Image, ImageDraw, ImageFont
from chunking import Chunking

import torch
import torchvision
import torchvision.transforms as transforms

from detections import *
import subdetection as dc
from create_dataset import Create_dataset
from frames_for_specific_scale import Frames_for_specific_scale
from image_with_detections import Image_with_detections



def try_chunking():
    image = Image.open("shrinked3_IMGP0017.jpg")
    image = image.convert('L')  # convert into grayscale
    Chunking.init(image)
    (scaled_chunks, scaled_positions) = Chunking.get_chunks_of_image_at_different_scales(strides=(5, 5), nb_shrinkages=3, division_factor=2)
    print(scaled_positions)
    for i in range(len(scaled_chunks)):
        l = len(scaled_chunks[i])
        print("number of positions:", l)
        # for j in range(l):
        # scaled_chunks[i][j].save("crops_visualization/cropped" + str(i) + "-" + str(j) + ".JPG", "JPEG") # just to see the result


def try_create_dataset():
    # We have a list of Images (here, we create 3 images)
    image = Image.open("IMGP0017.JPG")
    # Convert into grayscale
    image = image.convert("L")
    # We have a list of Images
    images = [image.crop((0, 0, 1000, 800)), image.crop((1000, 800, 2000, 1600)), image.crop((2000, 1600, 3000, 2400))]

    # We define a transformation to apply to dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    # We get the dataset from these Images
    ds = Create_dataset(images, transform)

    # We get the dataloader. Minibatches of 2
    dataloader = torch.utils.data.DataLoader(ds, batch_size=2,
                                             shuffle=True, num_workers=2)

    # What are really stored in the dataloader we've created ?
    # We can see the content of a minibatch!! Here, since we have 3 images,
    # the second minibatch only contains 1 image.
    for data in dataloader:
        images, labels = data
        print("images:", images)
        print("images size:", images.size())
        print("labels:", labels)
        print("labels size:", labels.size())


# After the NN
def try_from_scores_saving_detections():

    """1/ Let we have these Frames_for_specific_scale instances,
    so we have the frames at different scales with positions and refined scores associated:
    """
    # For each image size, we won't add all the possible frames, it will be too long. But it's not important to test.
    # We don't need the dataset to test, this was for the NN step

    # Will be done before NN feeding
    frames_for_scale_1 = Frames_for_specific_scale(1, [], [(0, 0), (95, 130), (100, 130), (500, 300)])
    frames_for_scale_2 = Frames_for_specific_scale(1.2, [], [(0, 0), (81, 105), (333, 458), (500, 300)])
    frames_for_scale_3 = Frames_for_specific_scale(2.4, [], [(0, 1), (38, 53), (167, 230), (312, 33), (400, 20)])

    # Will be while NN feeding. This is the 3rd step. TODO; implement set_scores
    frames_for_scale_1.scores = [-1, 0.6, 0.65, -0.4]
    frames_for_scale_2.scores = [-1, 0.7, 0.7, -0.4]
    frames_for_scale_3.scores = [-1.2, 0.53, 0.8, 0.9, -0.1]

    frames = []
    frames.append(frames_for_scale_1)
    frames.append(frames_for_scale_2)
    frames.append(frames_for_scale_3)


    """2/ Frome these, we get the subdetections."""
    subdetections = capture_subdetections(frames)
    print("subdetections:")
    for subdetection in subdetections:
        print(subdetection)

    """3/ Clustering of DetectionCandidates into Detections and filtering"""
    detections = get_detections(subdetections, min_samples=2)  # OK!
    print(detections)  # In the example, one subdetection is alon in a detection => this detection is discarded.


    """4/ Get the best candidates"""
    winners = get_best_clusters_candidates(subdetections, detections)
    for winner in winners:
        print(winner)

    """5/ Save an image with all subdetections, and then only with the kept subdetections"""

    # With all subdetections
    image = Image.open("catch_detec_images/blank_example.jpg")
    image_with_detections = Image_with_detections(image, subdetections)
    image_with_detections.save("catch_detec_images/all_detections.JPG")

    # With only winner subdetections
    image_with_detections.subdetections = winners
    image_with_detections.save("catch_detec_images/winner_detections.JPG")


if __name__ == "__main__":
    try_from_scores_saving_detections()