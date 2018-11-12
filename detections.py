import numpy as np
from sklearn.cluster import DBSCAN

#personal imports
from configuration import *
from subdetection import Subdetection


def capture_subdetections(frames_for_specific_scale):
    """
    :param frames_for_specific_scale: From this list, we build a list of DetectionCandidates. These are derived from
    the chunks whose associated score is =>0.
    :return: the list of DetectionCandidates
    """
    subdetections = []
    for frame in frames_for_specific_scale:
        for i in range(len(frame.scores)):
            if frame.scores[i] >= threshold_score:
                new_dimensions = (int(36 * frame.scaling_factor), int(36 * frame.scaling_factor))
                """To get the position in the rescaled image (recall that it is top-left), 
                we first scale linearly the center, and then from the center, we deduced the new position
                """
                old_center = get_center_of_frame(
                    frame.positions[i])  # center in the scaled image (so before rescaling to original size)
                new_x = old_center[0] * frame.scaling_factor
                new_x = int(new_x - new_dimensions[0] / 2)
                new_y = old_center[1] * frame.scaling_factor
                new_y = int(new_y - new_dimensions[1] / 2)
                subdetections.append(Subdetection(frame.scores[i], (new_x, new_y), new_dimensions))
    return subdetections


def get_best_clusters_candidates(subdetections, clusters):
    """
    :param subdetections: a list of DetectionCandidates
    :param clusters: a list of clusters
    :return: the chosen candidate for each cluster
    => we have a list of DetectionCandidates. Each one represents the Detection that contains it.
    """
    chosen = []
    best_score = -1000
    best_candidate_index = -1000
    for c in clusters:
        best_score = -1000
        best_candidate_index = -1000
        for ind in c:  # indices of our list of subdetections
            if best_score < subdetections[ind].score:
                best_score = subdetections[ind].score
                best_candidate_index = ind
        chosen.append(subdetections[best_candidate_index])
    return chosen


def get_center_of_frame(frame_position):
    """Returns the center associated to frame_position in the scaled image
    (so the size of a frame is not yet resized:36x36)
    """
    return (frame_position[0] + 18, frame_position[1] + 18)


def get_clusters_from_frames(centers_frames, eps=3, min_samples=1):
    """
    :param eps: maximum distance between two samples
    :param min_samples: number of samples in a neighborhood for a point to be a core point
    :param centers_frames: List of coordinates of the center of frames (x,y)
    :return: A list of lists --> for one list, there is the indexes of the frames
    We don't have an explicit class to design a Cluster, which is a list of (subdetection) indices.
    """
    print("inside get_clusters_from_frames")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_frames)
    dict_clusters = dict()
    for index, value in enumerate(clustering.labels_):
        # If -1, it means the subdetection does not have enough neighbor (point is outlier)
        # => we ignore it and don't build a cluster from it.
        if value != -1:
            if value not in dict_clusters:
                dict_clusters[value] = [index]
            else:
                dict_clusters[value].append(index)
    return list(dict_clusters.values())


def get_detections(subdetections, min_samples=1):
    """
    :param subdetections: all the subdetections we have to be clustered
    :param min_samples: number of minimum subdetections in the detection for the detection to be accepted. Otherwise,
    the detection is rejected => not added to the returned list of detections.
    => With this function, we both do the steps of getting the detections and filtering them (discarding/rejecting those
    with too few subdetections)!
    :return: list of clusters of subdetections
    """
    centers_frames = []
    for subdetect in subdetections:
        center = subdetect.get_center()
        centers_frames.append([center[0], center[1]])  # because DBSCAN works with vectors that are lists, not tuples
    print("inside get_detections")
    return get_clusters_from_frames(centers_frames, 50, min_samples) # TODO: eps must be proportional to the image dimensions

