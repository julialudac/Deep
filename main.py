from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#personal imports
from configuration import *
from create_dataset import Create_dataset
from chunking import Chunking
from detections import *
from frames_for_specific_scale import Frames_for_specific_scale
from image_with_detections import Image_with_detections
import network


def train_neural_network(train_loader, net, optimizer, criterion):
    """
    :param train_loader:
    :param net: NN to train
    :param optimizer:
    :param criterion:
    :return: NN
    """
    start_time = time.time()
    print('Start training')
    for epoch in range(1):  # loop over the dataset multiple times

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
    print("time spent:", time.time() - start_time)
    return net


def test_neural_network_original_dataset(test_loader, net):
    """
    :param test_loader:
    :param net: NN already trained
    :return: None
    """
    start_time = time.time()
    print('Start testing')
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    classes = (0, 1,)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels = torch.LongTensor(([1 if label > 1 else label for label in labels]))
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
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
            labels = torch.LongTensor(([1 if label > 1 else label for label in labels]))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    print('Finished Training')
    print("time spent:", time.time() - start_time)


def test_neural_network_own_dataset(path_to_image, net):
    """
    :param path_to_image: string which indicates where our image is stored
    :param net: NN already trained
    :return: None
    """
    start_time = time.time()
    print('Start testing')

    image = Image.open(path_to_image)
    # Convert into grayscale
    # image = image.convert("L")

    # cut image to different scales
    Chunking.init(image)
    (scaled_images, scaled_positions) = Chunking.get_chunks_of_image_at_different_scales(strides=strides, nb_shrinkages=3, division_factor=2)

    # compute score for each frame
    frames = []
    scaled_factor = 1
    for index in range(len(scaled_images)):
        dataset = Create_dataset(scaled_images[index])
        # get the dataloader. Minibatches of 4
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                                 shuffle=True, num_workers=2)
        scores = []
        with torch.no_grad():
            for data in dataloader:
                images, _ = data
                outputs = net(images)  # [(score of class_0, score of class_1), ...]
                outputs = outputs.numpy()
                outputs = outputs.tolist()
                scores.extend(outputs)
        frames.append(Frames_for_specific_scale(scaled_factor, dataset, scaled_positions[index], scores))
        scaled_factor = division_factor * (2 ** index)

    # subdetections
    subdetections = capture_subdetections(frames)

    # clustering of Subdetections into clusters_of_subdetections and filtering
    # min_samples --> number of min detection to confirm the detection
    clusters_of_subdetections = get_detections(subdetections, min_samples=min_samples)
    print("clusters of subdetections")
    print(clusters_of_subdetections)  # In the example, one subdetection is alone in a detection => this detection is discarded.

    # get the best candidates
    winners = get_best_clusters_candidates(subdetections, clusters_of_subdetections)
    for winner in winners:
        print(winner)

    # save an image with all subdetections, and then only with the kept subdetections
    image_with_detections = Image_with_detections(image, subdetections)
    original_name_image = path_to_image.split('.')

    # With all subdetections
    image_with_detections.save(original_name_image[0] + "_all_detections.JPG")

    # With only winner subdetections
    image_with_detections.subdetections = winners
    image_with_detections.save(original_name_image[0] + "_winners_detections.JPG")

    print('Finished Training')
    print("time spent:", time.time() - start_time)


if __name__ == "__main__":
    #training data set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_imagenet = torchvision.datasets.ImageFolder('start_deep/start_deep/train_images/',transform=transform)

    #data augmentation
    for theta in range(3):
        transform = transforms.Compose(
            [transforms.RandomRotation((90*(theta+1), 90*(theta+1))),
            transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        train_imagenet += torchvision.datasets.ImageFolder('start_deep/start_deep/train_images/',transform=transform)

    train_loader = torch.utils.data.DataLoader(train_imagenet, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    #define net and define optimizer
    net = network.Net()
    weights = torch.Tensor([1, 3])
    criterion = nn.CrossEntropyLoss(weights)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #training
    net = train_neural_network(train_loader, net, optimizer, criterion)

    while input("Do you want to test again? (yes/no)") == "yes":
        if input("Do you want to use the original testing dataset ? (yes/no)") == "yes" :
            #testing original data set (from teacher)
            test_imagenet = torchvision.datasets.ImageFolder('start_deep/start_deep/start_deep/test_images/',
                                                             transform=transform)
            test_loader = torch.utils.data.DataLoader(test_imagenet, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
            test_neural_network_original_dataset(test_loader, net)
        else:
            #testing any image
            path_to_image = input("(ex : own_dataset/ex3.JPG). The path to the image to test is :")
            test_neural_network_own_dataset(path_to_image, net)

