from PIL import Image
from cut_image import Chunking


def main():

    # Using cut_image
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









if __name__ == "__main__":
    main()