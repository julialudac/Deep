from PIL import Image
from cut_image import Chunking


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#personal imports
from CustomDatasetFromImages import CustomDatasetFromImages
import Net


def train_neural_network(train_loader, net, optimizer, criterion):
    print('Start training')
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return net


def test_neural_network(test_loader, net):
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    classes = (0, 1,)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def try_cut_image():
    im = Image.open("own_dataset/ex3.jpg")
    im = im.convert('L')  # convert into grayscale
    Chunking.init(im)
    (scaleschunks, scalespositions) = Chunking.get_imgchunks_atdiffscales(strides=(5, 5), nbshrinkages=3, divfactor=2)
    print(scalespositions)
    for i in range(len(scaleschunks)):
        l = len(scaleschunks[i])
        print("number of positions:", l)
        # for j in range(l):
        # scaleschunks[i][j].save("crops_visualization/cropped" + str(i) + "-" + str(j) + ".JPG", "JPEG") # just to see the result


def try_CustomDatasetFromImages(path_to_image):
    # We have a list of Images (here, we create 3 images)
    im = Image.open("own_dataset/ex3.JPG")
    # Convert into grayscale
    im = im.convert("L")
    # We have a list of Images
    ims = [im.crop((0, 0, 200, 200)), im.crop((200, 200, 400, 400)), im.crop((400, 400, 600, 600))]

    # We get the dataset from these Images
    ds = CustomDatasetFromImages(ims)

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

    return dataloader


if __name__ == "__main__":
    ###training data set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_imagenet = torchvision.datasets.ImageFolder('start_deep/start_deep/start_deep/train_images/',
                                                      transform=transform)
    train_loader = torch.utils.data.DataLoader(train_imagenet, batch_size=4, shuffle=True, num_workers=2)

    #teting data set
    test_imagenet = torchvision.datasets.ImageFolder('start_deep/start_deep/start_deep/test_images/',
                                                     transform=transform)
    test_loader = torch.utils.data.DataLoader(test_imagenet, batch_size=4, shuffle=True, num_workers=2)

    #define net and define optimizer
    net = Net.Net()
    weights = torch.Tensor([1, 0.5])
    criterion = nn.CrossEntropyLoss(weights)
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

    #training
    net = train_neural_network(train_loader, net, optimizer, criterion)

    while input("Do you want to test again? (yes/no)") == "yes":
        if input("Do you want to use the original training dataset ? (yes/no)") == "yes" :
            test_neural_network(test_loader, net)
        else:
            path_to_image = input("(ex : own_dataset/ex3.JPG). The path to the image to test is :")
            test_loader_small = try_CustomDatasetFromImages(path_to_image)
            test_neural_network(test_loader_small, net)
