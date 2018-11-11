from PIL import Image, ImageDraw, ImageFont
from cut_image import Chunking

import torch
import torchvision
import torchvision.transforms as transforms

import detections as det
import DetectionCandidate as dc
from CustomDatasetFromImages import CustomDatasetFromImages
from FramesAtGivenScaledImage import FramesAtGivenScaledImage
from ImageWithDetections import ImageWithDetections



def try_cut_image():
    im = Image.open("shrinked3_IMGP0017.jpg")
    im = im.convert('L')  # convert into grayscale
    Chunking.init(im)
    (scaleschunks, scalespositions) = Chunking.get_imgchunks_atdiffscales(strides=(5, 5), nbshrinkages=3, divfactor=2)
    print(scalespositions)
    for i in range(len(scaleschunks)):
        l = len(scaleschunks[i])
        print("number of positions:", l)
        # for j in range(l):
        # scaleschunks[i][j].save("crops_visualization/cropped" + str(i) + "-" + str(j) + ".JPG", "JPEG") # just to see the result


def try_CustomDatasetFromImages():
    # We have a list of Images (here, we create 3 images)
    im = Image.open("IMGP0017.JPG")
    # Convert into grayscale
    im = im.convert("L")
    # We have a list of Images
    ims = [im.crop((0, 0, 1000, 800)), im.crop((1000, 800, 2000, 1600)), im.crop((2000, 1600, 3000, 2400))]

    # We define a transformation to apply to dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    # We get the dataset from these Images
    ds = CustomDatasetFromImages(ims, transform)

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
def try_fromscores2savingdetections():

    """1/ Let we have these FramesAtGivenScaledImage instances,
    so we have the frames at different scales with positions and refined scores associated:
    """
    # For each image size, we won't add all the possible frames, it will be too long. But it's not important to test.
    # We don't need the dataset to test, this was for the NN step

    # Will be done before NN feeding
    fagsi1 = FramesAtGivenScaledImage(1, [], [(0, 0), (95, 130), (100, 130), (500, 300)])
    fagsi2 = FramesAtGivenScaledImage(1.2, [], [(0, 0), (81, 105), (333, 458), (500, 300)])
    fagsi3 = FramesAtGivenScaledImage(2.4, [], [(0, 1), (38, 53), (167, 230), (312, 33), (400, 20)])

    # Will be while NN feeding. This is the 3rd step. TODO; implement set_scores
    fagsi1.scores = [-1, 0.6, 0.65, -0.4]
    fagsi2.scores = [-1, 0.7, 0.7, -0.4]
    fagsi3.scores = [-1.2, 0.53, 0.8, 0.9, -0.1]

    framesAtGivenScaledImages = []
    framesAtGivenScaledImages.append(fagsi1)
    framesAtGivenScaledImages.append(fagsi2)
    framesAtGivenScaledImages.append(fagsi3)


    """2/ Frome these, we get the subdetections."""
    subdetections = dc.capture_good_positions(framesAtGivenScaledImages)
    print("subdetections:")
    for subd in subdetections:
        print(subd)
    # OK!


    """3/ Clustering of DetectionCandidates into Detections and filtering"""
    detections = det.getDetections(subdetections, min_samples=2)  # OK!
    print(detections)  # In the example, one subdetection is alon in a detection => this detection is discarded.


    """4/ Get the best candidates"""
    winners = det.get_best_clusters_candidates(subdetections, detections)
    for w in winners:
        print(w)

    """5/ Save an image with all subdetections, and then only with the kept subdetections"""

    # With all subdetections
    pilImage = Image.open("catch_detec_images/blank_example.jpg")
    imdet = ImageWithDetections(pilImage, subdetections)
    imdet.save("catch_detec_images/all_detections.JPG")

    # With only winner subdetections
    imdet.subdetections = winners
    imdet.save("catch_detec_images/winner_detections.JPG")



def main():
    try_fromscores2savingdetections()








if __name__ == "__main__":
    main()