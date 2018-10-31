from PIL import Image
from cut_image import Chunking

import torch
import torchvision
import torchvision.transforms as transforms
from CustomDatasetFromImages import CustomDatasetFromImages


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


def main():
    try_CustomDatasetFromImages()








if __name__ == "__main__":
    main()