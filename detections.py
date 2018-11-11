import numpy as np
from sklearn.cluster import DBSCAN

import configuration


def cluster_frames(centers_frames, eps=3, min_samples=1):
    """
    :param eps: maximum distance between two samples
    :param min_samples: number of samples in a neighborhood for a point to be a core point
    :param centers_frames: List of coordinates of the center of frames (x,y)
    :return: A list of lists --> for one list, there is the indexes of the frames
    We don't have an explicit class to design a Cluster, which is a list of (subdetection) indices.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_frames)
    dict_clusters = dict()
    for index, value in enumerate(clustering.labels_):
        # If -1, it means the subdetection does not have enough neighbor => we ignore it and don't build a cluster from it.
        if value != -1:
            if value not in dict_clusters:
                dict_clusters[value] = [index]
            else:
                dict_clusters[value].append(index)
    return list(dict_clusters.values())


def getDetections(subdetections, min_samples=1):
    """
    :param subdetections: all the subdetections we have to be clustered
    :param min_samples: number of minimum subdetections in the detection for the detection to be accepted. Otherwise,
    the detection is rejected => not added to the returned list of detections.
    => With this function, we both do the steps of getting the detections and filtering them (discarding/rejecting those
    with too few subdetections)!
    :return: list of Detections
    """
    centers_frames = []
    for subd in subdetections:
        center = subd.get_center()
        centers_frames.append([center[0], center[1]])  # because DBSCAN works with vectors that are lists, not tuples
    return cluster_frames(centers_frames, (configuration.image_dims[0]+configuration.image_dims[1])/40, min_samples)

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
